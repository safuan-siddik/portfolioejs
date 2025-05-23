const express = require("express");
const router = express.Router();
const bcrypt = require("bcryptjs");
const nodemailer = require("nodemailer");
const { body, validationResult } = require("express-validator");
const expressSanitizer = require("express-sanitizer");
const mysql = require("mysql2");
const config = require("../config");

// Create a promise-based pool
const pool = mysql
  .createPool({
    ...config.database,
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0,
  })
  .promise();

// Helper function for database queries
const query = async (sql, params) => {
  try {
    const [results] = await pool.query(sql, params);
    return results;
  } catch (error) {
    console.error("Database query error:", error);
    throw error;
  }
};

// Middleware to check if a user is logged in
const isAuthenticated = (req, res, next) => {
  if (req.session.userId) {
    next();
  } else {
    req.session.returnTo = req.originalUrl;
    res.redirect("/login");
  }
};

// Email configuration with better error handling
const transporter = nodemailer.createTransport({
  host: "smtp.ethereal.email",
  port: 587,
  auth: {
    user: "veda.schmitt@ethereal.email",
    pass: "pHmpPaQcuGM92bgckD",
  },
  secure: false,
  tls: {
    rejectUnauthorized: false,
  },
});

// Validation middleware
const validateContact = [
  body("name").trim().notEmpty().withMessage("Name is required").escape(),
  body("email")
    .trim()
    .isEmail()
    .withMessage("Please enter a valid email")
    .normalizeEmail(),
  body("subject").trim().notEmpty().withMessage("Subject is required").escape(),
  body("message").trim().notEmpty().withMessage("Message is required").escape(),
];

const validateSignup = [
  body("username")
    .trim()
    .notEmpty()
    .withMessage("Username is required")
    .isLength({ min: 3, max: 20 })
    .withMessage("Username must be between 3 and 20 characters")
    .matches(/^[a-zA-Z0-9_]+$/)
    .withMessage("Username can only contain letters, numbers, and underscores")
    .escape(),
  body("email")
    .trim()
    .isEmail()
    .withMessage("Please enter a valid email address")
    .normalizeEmail(),
  body("password")
    .trim()
    .isLength({ min: 8 })
    .withMessage("Password must be at least 8 characters long")
    .matches(/^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$/)
    .withMessage(
      "Password must contain at least one letter, one number, and one special character"
    )
    .escape(),
];

// Route Handlers

// Homepage
router.get("/", (req, res) => {
  res.render("index", {
    title: "Home",
    description: "Welcome to my portfolio website",
  });
});

// About page
router.get("/about", (req, res) => {
  res.render("about", {
    title: "About Me",
    description: "Learn more about my background and skills",
  });
});

// Work page
router.get("/work", (req, res) => {
  res.render("work", {
    title: "My Work",
    description: "Explore my projects and portfolio",
  });
});

// Contact page
router.get("/contact", (req, res) => {
  res.render("contact", {
    title: "Contact Me",
    description: "Get in touch with me",
  });
});

// Contact Form Submission
router.post("/contact", validateContact, async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).render("contact", {
        errors: errors.array(),
        data: req.body,
        title: "Contact Me",
        description: "Get in touch with me",
      });
    }

    const { name, email, subject, message } = req.body;

    const mailOptions = {
      from: email,
      to: "demixelord@gmail.com",
      subject: subject || "Contact Form Submission",
      text: `
        Name: ${name}
        Email: ${email}
        Subject: ${subject}
        Message: ${message}
      `,
      html: `
        <h2>New Contact Form Submission</h2>
        <p><strong>Name:</strong> ${name}</p>
        <p><strong>Email:</strong> ${email}</p>
        <p><strong>Subject:</strong> ${subject}</p>
        <p><strong>Message:</strong></p>
        <p>${message}</p>
      `,
    };

    await transporter.sendMail(mailOptions);
    req.flash("success", "Your message has been sent successfully!");
    res.redirect("/contact");
  } catch (error) {
    console.error("Error sending email:", error);
    req.flash("error", "Error sending message. Please try again later.");
    res.redirect("/contact");
  }
});

// Sign-Up Page
router.get("/signup", (req, res) => {
  if (req.session.userId) {
    return res.redirect("/");
  }
  res.render("signup", {
    title: "Sign Up",
    description: "Create a new account",
  });
});

// Handle Sign-Up
router.post("/signup", validateSignup, async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).render("signup", {
        errors: errors.array(),
        data: req.body,
        title: "Sign Up",
        description: "Create a new account",
      });
    }

    const { username, email, password } = req.body;

    // Check if username or email already exists
    const [existingUsers] = await query(
      "SELECT * FROM Users WHERE username = ? OR email = ?",
      [username, email]
    );

    if (existingUsers.length > 0) {
      return res.status(400).render("signup", {
        errors: [{ msg: "Username or email already exists" }],
        data: req.body,
        title: "Sign Up",
        description: "Create a new account",
      });
    }

    const hashedPassword = await bcrypt.hash(password, 12);
    await query(
      "INSERT INTO Users (username, email, password_hash) VALUES (?, ?, ?)",
      [username, email, hashedPassword]
    );

    req.flash("success", "Account created successfully! Please log in.");
    res.redirect("/login");
  } catch (error) {
    console.error("Signup error:", error);
    req.flash("error", "Error creating account. Please try again.");
    res.redirect("/signup");
  }
});

// Login page
router.get("/login", (req, res) => {
  if (req.session.userId) {
    return res.redirect("/");
  }
  res.render("login", {
    title: "Login",
    description: "Sign in to your account",
  });
});

// Handle Login
router.post(
  "/login",
  [
    body("email")
      .trim()
      .isEmail()
      .withMessage("Please enter a valid email")
      .normalizeEmail(),
    body("password").trim().notEmpty().withMessage("Password is required"),
  ],
  async (req, res) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).render("login", {
          errors: errors.array(),
          data: req.body,
          title: "Login",
          description: "Sign in to your account",
        });
      }

      const { email, password } = req.body;
      console.log("Attempting login for email:", email);

      // Get user from database
      const users = await query("SELECT * FROM Users WHERE email = ?", [email]);
      console.log("Database query result:", users);

      if (!users || users.length === 0) {
        console.log("No user found for email:", email);
        req.flash("error", "Invalid email or password");
        return res.status(401).render("login", {
          data: req.body,
          title: "Login",
          description: "Sign in to your account",
        });
      }

      const user = users[0];
      console.log("Found user:", {
        id: user.id,
        email: user.email,
        hasPasswordHash: !!user.password_hash,
      });

      // Check if user has a password hash
      if (!user.password_hash) {
        console.error("User found but no password hash:", user);
        req.flash("error", "Account error. Please contact support.");
        return res.status(500).render("login", {
          data: req.body,
          title: "Login",
          description: "Sign in to your account",
        });
      }

      // Verify password
      const isValidPassword = await bcrypt.compare(
        password,
        user.password_hash
      );
      console.log("Password validation result:", isValidPassword);

      if (!isValidPassword) {
        console.log("Invalid password for user:", email);
        req.flash("error", "Invalid email or password");
        return res.status(401).render("login", {
          data: req.body,
          title: "Login",
          description: "Sign in to your account",
        });
      }

      // Set user session
      req.session.userId = user.id;
      req.session.username = user.username;
      console.log("Login successful for user:", user.username);

      // Redirect to the page they were trying to access or home
      const returnTo = req.session.returnTo || "/";
      delete req.session.returnTo;
      res.redirect(returnTo);
    } catch (error) {
      console.error("Login error details:", {
        message: error.message,
        stack: error.stack,
        query: error.sqlMessage,
      });
      req.flash("error", "Error during login. Please try again.");
      res.redirect("/login");
    }
  }
);

// Logout
router.get("/logout", (req, res) => {
  req.session.destroy((err) => {
    if (err) {
      console.error("Logout error:", err);
      return res.redirect("/");
    }
    res.clearCookie("connect.sid");
    res.redirect("/");
  });
});

// Forum page
router.get("/forum", isAuthenticated, async (req, res) => {
  try {
    console.log("Fetching forum posts...");
    const posts = await query(`
      SELECT Posts.*, Users.username, 
             DATE_FORMAT(Posts.created_at, '%Y-%m-%d %H:%i') as formatted_date
      FROM Posts
      JOIN Users ON Posts.user_id = Users.id
      ORDER BY Posts.created_at DESC
    `);
    console.log("Query result:", posts);

    // Check if posts table exists
    const tables = await query("SHOW TABLES LIKE 'Posts'");
    console.log("Posts table exists:", tables.length > 0);

    // Check if there are any posts
    const postCount = await query("SELECT COUNT(*) as count FROM Posts");
    console.log("Total posts in database:", postCount[0]?.count || 0);

    res.render("forum", {
      posts: posts || [], // Ensure posts is always an array
      title: "Forum",
      description: "Discussion forum",
      csrfToken: req.csrfToken(), // Add CSRF token
    });
  } catch (error) {
    console.error("Forum error:", error);
    req.flash("error", "Error loading forum posts.");
    res.redirect("/");
  }
});

// Handle new post
router.post(
  "/forum",
  isAuthenticated,
  [
    body("content")
      .trim()
      .notEmpty()
      .withMessage("Post content is required")
      .isLength({ max: 1000 })
      .withMessage("Post must be less than 1000 characters")
      .escape(),
  ],
  async (req, res) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        const posts = await query(`
          SELECT Posts.*, Users.username,
                 DATE_FORMAT(Posts.created_at, '%Y-%m-%d %H:%i') as formatted_date
          FROM Posts
          JOIN Users ON Posts.user_id = Users.id
          ORDER BY Posts.created_at DESC
        `);

        return res.status(400).render("forum", {
          errors: errors.array(),
          posts: posts || [], // Ensure posts is always an array
          title: "Forum",
          description: "Discussion forum",
        });
      }

      await query("INSERT INTO Posts (user_id, content) VALUES (?, ?)", [
        req.session.userId,
        req.body.content,
      ]);

      req.flash("success", "Post created successfully!");
      res.redirect("/forum");
    } catch (error) {
      console.error("Post creation error:", error);
      req.flash("error", "Error creating post. Please try again.");
      res.redirect("/forum");
    }
  }
);

// Search function
router.get("/search", isAuthenticated, async (req, res) => {
  try {
    const { query: searchQuery } = req.query; // Get the search term from the query string
    if (!searchQuery) {
      return res.redirect("/forum");
    }

    console.log("Inside /search route handler.");
    console.log("Value of 'query' variable:", typeof searchQuery, searchQuery);
    console.log("Value of 'pool' variable:", typeof pool, pool);

    const posts = await query(
      `
      SELECT Posts.id, Users.username,
             DATE_FORMAT(Posts.created_at, '%Y-%m-%d %H:%i') as formatted_date,
             Posts.content
      FROM Posts
      JOIN Users ON Posts.user_id = Users.id
      WHERE Posts.content LIKE ?
      ORDER BY Posts.created_at DESC
    `,
      [`%${searchQuery}%`]
    );

    res.render("forum", {
      posts: posts || [], // Ensure posts is always an array
      searchQuery: searchQuery,
      title: "Search Results",
      description: `Search results for: ${searchQuery}`,
      csrfToken: req.csrfToken(), // Add CSRF token
    });
  } catch (error) {
    console.error("Search error:", error);
    req.flash("error", "Error performing search.");
    res.redirect("/forum");
  }
});

// Trading Bot Project page
router.get("/trading-bot", (req, res) => {
  res.render("trading-bot");
});

// Trading Bot API endpoints
router.get("/api/trading-bot/backtest", (req, res) => {
  // Here we'll add the backtest results endpoint
  res.json({
    success: true,
    data: {
      totalTrades: 150,
      winRate: 0.65,
      profitFactor: 1.8,
      sharpeRatio: 1.5,
      maxDrawdown: 0.15,
      annualReturn: 0.25,
    },
  });
});

// API Routes

// API Route to Fetch All Posts (GET /api/posts)
router.get("/api/posts", async (req, res) => {
  try {
    const posts = await query(
      `
      SELECT Posts.id, Posts.content, Users.username, Posts.created_at
      FROM Posts
      JOIN Users ON Posts.user_id = Users.id
      ORDER BY Posts.created_at DESC
    `
    );

    // Return the posts as JSON
    res.json(posts);
  } catch (error) {
    console.error("Error fetching API posts:", error);
    res.status(500).json({ message: "Error loading posts." });
  }
});

// API Route for Searching Posts (GET /api/search)
router.get("/api/search", async (req, res) => {
  const { query: searchQuery } = req.query; // Get the search term from the query string
  if (!searchQuery) {
    return res.status(400).json({ message: "No search query provided." });
  }

  try {
    // SQL query to search posts that contain the query term in the content
    const posts = await query(
      `
      SELECT Posts.id, Posts.content, Users.username, Posts.created_at
      FROM Posts
      JOIN Users ON Posts.user_id = Users.id
      WHERE Posts.content LIKE ?
      ORDER BY Posts.created_at DESC
    `,
      [`%${searchQuery}%`]
    );

    // Return the search results as JSON
    res.json(posts);
  } catch (error) {
    console.error("Error searching API posts:", error);
    res.status(500).json({ message: "Error occurred while searching posts." });
  }
});

// Export the router object so it can be used in `app.js`
module.exports = router;
