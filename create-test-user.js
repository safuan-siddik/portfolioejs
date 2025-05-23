const mysql = require("mysql2/promise");
const bcrypt = require("bcryptjs");
const config = require("./config");

async function createTestUser() {
  let pool;
  try {
    // Create connection pool
    pool = mysql.createPool({
      ...config.database,
      waitForConnections: true,
      connectionLimit: 10,
      queueLimit: 0,
    });

    // Test database connection
    const connection = await pool.getConnection();
    console.log("Successfully connected to database");
    connection.release();

    // Check if test user exists
    const [users] = await pool.query(
      "SELECT * FROM Users WHERE email = ? OR username = ?",
      ["test@example.com", "testuser"]
    );

    if (users.length === 0) {
      // Create test user
      const hashedPassword = await bcrypt.hash("Test123!", 12);
      const [result] = await pool.query(
        "INSERT INTO Users (username, email, password_hash) VALUES (?, ?, ?)",
        ["testuser", "test@example.com", hashedPassword]
      );
      console.log("Test user created successfully");

      // Create a test post
      await pool.query("INSERT INTO Posts (user_id, content) VALUES (?, ?)", [
        result.insertId,
        "This is a test post to verify forum functionality!",
      ]);
      console.log("Test post created successfully");
    } else {
      console.log("Test user already exists");

      // Check if test post exists
      const [posts] = await pool.query(
        "SELECT * FROM Posts WHERE user_id = ?",
        [users[0].id]
      );

      if (posts.length === 0) {
        // Create a test post
        await pool.query("INSERT INTO Posts (user_id, content) VALUES (?, ?)", [
          users[0].id,
          "This is a test post to verify forum functionality!",
        ]);
        console.log("Test post created successfully");
      } else {
        console.log("Test post already exists");
      }
    }
  } catch (error) {
    console.error("Error:", error);
    if (error.code === "ECONNREFUSED") {
      console.error(
        "Could not connect to database. Please ensure MySQL is running."
      );
    } else if (error.code === "ER_ACCESS_DENIED_ERROR") {
      console.error("Access denied. Please check your database credentials.");
    } else if (error.code === "ER_BAD_DB_ERROR") {
      console.error("Database does not exist. Please run database.sql first.");
    }
  } finally {
    if (pool) {
      await pool.end();
    }
  }
}

createTestUser();
