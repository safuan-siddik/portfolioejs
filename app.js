const express = require("express");
const ejs = require("ejs");
const bodyParser = require("body-parser");
const mysql = require("mysql2");
const session = require("express-session");
const path = require("path");
const expressSanitizer = require("express-sanitizer");
const helmet = require("helmet");
const rateLimit = require("express-rate-limit");
const config = require("./config");
const compression = require("compression");
const cookieParser = require("cookie-parser");
const csrf = require("csurf");
const flash = require("connect-flash");
const MySQLStore = require("express-mysql-session")(session);

const app = express();

app.set("trust proxy", true);

// Security middleware
app.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'", "'unsafe-inline'", "cdn.jsdelivr.net"],
        styleSrc: ["'self'", "'unsafe-inline'", "fonts.googleapis.com"],
        fontSrc: ["'self'", "fonts.gstatic.com"],
        imgSrc: ["'self'", "data:", "blob:"],
        connectSrc: ["'self'"],
      },
    },
    crossOriginEmbedderPolicy: false,
  })
);

// Performance and security middleware
app.use(compression());
app.use(cookieParser());
app.use(expressSanitizer());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.json());
app.use(
  express.static(path.join(__dirname, "public"), {
    maxAge: "1d",
    etag: true,
  })
);

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: "Too many requests from this IP, please try again later.",
});
app.use(limiter);

// Session configuration
const sessionStore = new MySQLStore({
  ...config.database,
  createDatabaseTable: true,
});

app.use(
  session({
    secret: config.sessionSecret,
    resave: false,
    saveUninitialized: false,
    store: sessionStore,
    cookie: {
      secure: process.env.NODE_ENV === "production",
      maxAge: 1000 * 60 * 60 * 24, // 24 hours
    },
  })
);

// CSRF protection
app.use(csrf());

// Flash messages
app.use(flash());

// Global middleware
app.use((req, res, next) => {
  res.locals.csrfToken = req.csrfToken();
  res.locals.error = req.flash("error");
  res.locals.success = req.flash("success");
  res.locals.user = req.session.userId ? true : false;
  res.locals.currentPath = req.path;
  next();
});

// Set EJS as the view engine
app.set("view engine", "ejs");

// MySQL Connection Pool with better error handling
const pool = mysql.createPool({
  ...config.database,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
  enableKeepAlive: true,
  keepAliveInitialDelay: 0,
});

// Test database connection with retry logic
const connectWithRetry = async (retries = 5, delay = 5000) => {
  try {
    const connection = await pool.promise().getConnection();
    console.log("Successfully connected to MySQL database");

    // Test the connection by running a simple query
    await connection.query("SELECT 1");
    console.log("Database connection test successful");

    connection.release();
    return true;
  } catch (err) {
    console.error("Error connecting to the database:", err);
    if (retries > 0) {
      console.log(
        `Retrying connection in ${
          delay / 1000
        } seconds... (${retries} attempts remaining)`
      );
      await new Promise((resolve) => setTimeout(resolve, delay));
      return connectWithRetry(retries - 1, delay);
    } else {
      console.error("Failed to connect to database after multiple retries");
      process.exit(1);
    }
  }
};

// Initialize database connection
(async () => {
  try {
    await connectWithRetry();

    // Create tables if they don't exist
    const createUsersTable = `
      CREATE TABLE IF NOT EXISTS Users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(50) NOT NULL UNIQUE,
        email VARCHAR(100) NOT NULL UNIQUE,
        password_hash VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
      )
    `;

    const createPostsTable = `
      CREATE TABLE IF NOT EXISTS Posts (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES Users(id) ON DELETE CASCADE
      )
    `;

    // Execute table creation queries separately
    await pool.promise().query(createUsersTable);
    console.log("Users table verified/created successfully");

    await pool.promise().query(createPostsTable);
    console.log("Posts table verified/created successfully");
  } catch (err) {
    console.error("Error initializing database:", err);
    process.exit(1);
  }
})();

// Attach the database pool to the app
app.set("db", pool);

// Load routes BEFORE error handlers
const mainRoutes = require("./routes/main");
app.use("/", mainRoutes);

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);

  // Handle CSRF token errors
  if (err.code === "EBADCSRFTOKEN") {
    return res.status(403).render("error", {
      title: "Form Submission Error",
      message: "Invalid form submission. Please try again.",
      error: process.env.NODE_ENV === "development" ? err : {},
    });
  }

  // Handle other errors
  res.status(err.status || 500).render("error", {
    title: "Error",
    message: err.message || "Something went wrong!",
    error: process.env.NODE_ENV === "development" ? err : {},
  });
});

// 404 handler - must be after all other routes
app.use((req, res) => {
  res.status(404).render("error", {
    title: "404 Not Found",
    message: "The page you're looking for doesn't exist",
    error: {},
  });
});

// Start the server
const server = app.listen(config.port, () => {
  console.log(`Server is running on port ${config.port}`);
});

// Graceful shutdown
process.on("SIGTERM", () => {
  console.log("SIGTERM signal received: closing HTTP server");
  server.close(() => {
    console.log("HTTP server closed");
    pool.end((err) => {
      if (err) {
        console.error("Error closing database pool:", err);
      }
      console.log("Database pool closed");
      process.exit(0);
    });
  });
});
