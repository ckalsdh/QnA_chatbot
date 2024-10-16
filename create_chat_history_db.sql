-- Create database
CREATE DATABASE chat_history_db;

-- Use the database
USE chat_history_db;

-- Create chat_history table
CREATE TABLE chat_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255),
    timestamp DATETIME,
    sender VARCHAR(10),
    message TEXT
);

