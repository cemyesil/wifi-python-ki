import sqlite3
from pathlib import Path

DB_PATH = Path("business.db")

DDL = """
CREATE TABLE IF NOT EXISTS clients (
  id     INTEGER PRIMARY KEY,
  name   TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS bookings (
  id            INTEGER PRIMARY KEY,
  client_id     INTEGER NOT NULL,
  booking_date  TEXT NOT NULL,               -- ISO date 'YYYY-MM-DD'
  amount        REAL NOT NULL,               -- revenue for this booking
  notes         TEXT,
  FOREIGN KEY (client_id) REFERENCES clients(id)
);

CREATE INDEX IF NOT EXISTS idx_bookings_client_date
  ON bookings(client_id, booking_date);

CREATE INDEX IF NOT EXISTS idx_bookings_year
  ON bookings(substr(booking_date,1,4));
"""

SAMPLE_DATA = [
    ("Google", [
        ("2025-01-15", 12000.00, "Q1 kickoff"),
        ("2025-06-02", 18000.50, "Mid-year push"),
        ("2024-11-30", 9000.00,  "Black Friday"),
    ]),
    ("Acme Corp", [
        ("2025-02-10", 5000.00,  "Pilot"),
        ("2025-09-01", 7500.00,  "Renewal"),
    ]),
]

def init_db():
    first_time = not DB_PATH.exists()
    con = sqlite3.connect(DB_PATH)
    try:
        con.executescript(DDL)

        if first_time:
            cur = con.cursor()
            for client, rows in SAMPLE_DATA:
                cur.execute("INSERT INTO clients(name) VALUES (?)", (client,))
                client_id = cur.lastrowid
                cur.executemany(
                    "INSERT INTO bookings(client_id, booking_date, amount, notes) VALUES (?,?,?,?)",
                    [(client_id, d, a, n) for (d, a, n) in rows]
                )
            con.commit()
    finally:
        con.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized at", DB_PATH)
