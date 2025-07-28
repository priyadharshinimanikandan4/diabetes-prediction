import sqlite3

conn = sqlite3.connect("database/diabetes.db")
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    bmi REAL,
    glucose REAL,
    bp REAL,
    insulin REAL,
    prediction TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)''')

conn.commit()
conn.close()
print("Database initialized successfully!")
