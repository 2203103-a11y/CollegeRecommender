import sqlite3

conn = sqlite3.connect('user_data.db')  # Creates database file
c = conn.cursor()

# Create table for storing user details
c.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                phone TEXT,
                gender TEXT
            )''')

conn.commit()
conn.close()
