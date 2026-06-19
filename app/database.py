import pyodbc
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import date
from urllib.parse import quote_plus

SERVER   = r"LAPTOP-VV7US68V\SQLEXPRESS"
DATABASE = "attendance_db"

CONNECTION_STRING = (
    "DRIVER={SQL Server};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    "Trusted_Connection=yes;"
)

engine = create_engine(
    "mssql+pyodbc:///?odbc_connect=" + quote_plus(CONNECTION_STRING)
)

def test_connection():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT @@VERSION")).fetchone()
        print("Connected:", result[0][:80])

def log_attendance(person_id, name, track_id):
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO attendance_log (person_id, name, date, track_id)
            VALUES (:pid, :name, CONVERT(date, :date, 23), :tid)
        """), {"pid": int(person_id),
               "name": str(name),
               "date": str(date.today()),
               "tid": int(track_id)})
        conn.commit()

def get_daily_attendance(target_date=None):
    if target_date is None:
        target_date = date.today()
    d = str(target_date)
    with engine.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT
                a.name,
                p.department,
                MIN(a.timestamp) as first_seen,
                COUNT(*) as frames
            FROM attendance_log a
            JOIN persons p ON a.person_id = p.id
            WHERE a.date = '{d}'
            GROUP BY a.person_id, a.name, p.department
        """), conn)
    return df

def get_monthly_attendance(year, month):
    y, m = int(year), int(month)
    with engine.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT name, date, COUNT(*) as count
            FROM attendance_log
            WHERE YEAR(date) = {y} AND MONTH(date) = {m}
            GROUP BY name, date
            ORDER BY date
        """), conn)
    return df

def get_hourly_breakdown(target_date=None):
    if target_date is None:
        target_date = date.today()
    d = str(target_date)
    with engine.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT
                DATEPART(HOUR, timestamp) as hour,
                COUNT(DISTINCT person_id) as count
            FROM attendance_log
            WHERE date = '{d}'
            GROUP BY DATEPART(HOUR, timestamp)
            ORDER BY hour
        """), conn)
    return df

def get_absent_today(target_date=None):
    """Returns list of persons NOT present today"""
    if target_date is None:
        target_date = date.today()
    d = str(target_date)
    with engine.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT name, employee_id, department
            FROM persons
            WHERE id NOT IN (
                SELECT DISTINCT person_id
                FROM attendance_log
                WHERE date = '{d}'
            )
        """), conn)
    return df

def get_attendance_percentage(year, month):
    """Returns attendance % per person for a given month"""
    y, m = int(year), int(month)
    with engine.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT
                p.name,
                p.department,
                COUNT(DISTINCT a.date) as days_present,
                (
                    SELECT COUNT(DISTINCT date)
                    FROM attendance_log
                    WHERE YEAR(date) = {y} AND MONTH(date) = {m}
                ) as total_days,
                CAST(COUNT(DISTINCT a.date) AS FLOAT) * 100.0 /
                NULLIF((
                    SELECT COUNT(DISTINCT date)
                    FROM attendance_log
                    WHERE YEAR(date) = {y} AND MONTH(date) = {m}
                ), 0) as percentage
            FROM persons p
            LEFT JOIN attendance_log a
                ON p.id = a.person_id
                AND YEAR(a.date) = {y}
                AND MONTH(a.date) = {m}
            GROUP BY p.id, p.name, p.department
            ORDER BY percentage DESC
        """), conn)
    return df

def get_late_arrivals(target_date=None, late_after="09:30"):
    """Returns persons who arrived after late_after time"""
    if target_date is None:
        target_date = date.today()
    d = str(target_date)
    with engine.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT
                a.name,
                p.department,
                MIN(a.timestamp) as arrival_time,
                CASE
                    WHEN CAST(MIN(a.timestamp) AS TIME) > '{late_after}'
                    THEN 'Late'
                    ELSE 'On Time'
                END as status
            FROM attendance_log a
            JOIN persons p ON a.person_id = p.id
            WHERE a.date = '{d}'
            GROUP BY a.person_id, a.name, p.department
            ORDER BY arrival_time
        """), conn)
    return df

def get_unknown_face_count(target_date=None):
    """Returns count of unknown face detections today"""
    if target_date is None:
        target_date = date.today()
    d = str(target_date)
    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT COUNT(*) FROM attendance_log
            WHERE date = '{d}' AND name = 'Unknown'
        """)).scalar()
    return result or 0

def get_weekly_summary():
    """Returns attendance count for last 7 days"""
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT
                date,
                COUNT(DISTINCT person_id) as present_count
            FROM attendance_log
            WHERE date >= CAST(DATEADD(DAY, -7, GETDATE()) AS DATE)
            GROUP BY date
            ORDER BY date
        """), conn)
    return df
    
def get_department_summary(target_date=None):
    if target_date is None:
        target_date = date.today()
    d = str(target_date)
    with engine.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT p.department, COUNT(DISTINCT a.person_id) as present
            FROM attendance_log a
            JOIN persons p ON a.person_id = p.id
            WHERE a.date = '{d}'
            GROUP BY p.department
        """), conn)
    return df

def get_person_monthly_summary(year, month):
    y, m = int(year), int(month)
    with engine.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT
                a.name,
                p.department,
                COUNT(DISTINCT a.date) as days_present,
                MIN(a.timestamp) as first_ever,
                AVG(CAST(DATEPART(HOUR, a.timestamp) AS FLOAT)
                    + CAST(DATEPART(MINUTE, a.timestamp) AS FLOAT) / 60.0
                ) as avg_arrival_hour
            FROM attendance_log a
            JOIN persons p ON a.person_id = p.id
            WHERE YEAR(a.date) = {y} AND MONTH(a.date) = {m}
            GROUP BY a.person_id, a.name, p.department
            ORDER BY days_present DESC
        """), conn)
    return df

def save_person_embedding(name, emp_id, department, embedding_bytes):
    with engine.connect() as conn:
        existing = conn.execute(
            text("SELECT id FROM persons WHERE employee_id = :eid"),
            {"eid": emp_id}
        ).fetchone()

        if existing:
            conn.execute(text("""
                UPDATE persons
                SET name       = :name,
                    department = :dept,
                    embedding  = :emb
                WHERE employee_id = :eid
            """), {"name": name, "eid": emp_id,
                   "dept": department, "emb": embedding_bytes})
            print(f"[DB] Updated existing record for employee_id={emp_id}")
        else:
            conn.execute(text("""
                INSERT INTO persons (name, employee_id, department, embedding)
                VALUES (:name, :eid, :dept, :emb)
            """), {"name": name, "eid": emp_id,
                   "dept": department, "emb": embedding_bytes})
            print(f"[DB] Inserted new record for {name}")

        conn.commit()

def load_all_embeddings():
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, name, embedding FROM persons"
        )).fetchall()
    return rows

def get_total_registered():
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM persons")).scalar()
    return result

def get_absent_today(target_date=None):
    """Returns list of persons NOT present today"""
    if target_date is None:
        target_date = date.today()
    d = str(target_date)
    with engine.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT name, employee_id, department
            FROM persons
            WHERE id NOT IN (
                SELECT DISTINCT person_id
                FROM attendance_log
                WHERE date = '{d}'
            )
        """), conn)
    return df

def get_attendance_percentage(year, month):
    """Returns attendance % per person for a given month"""
    y, m = int(year), int(month)
    with engine.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT
                p.name,
                p.department,
                COUNT(DISTINCT a.date) as days_present,
                (
                    SELECT COUNT(DISTINCT date)
                    FROM attendance_log
                    WHERE YEAR(date) = {y} AND MONTH(date) = {m}
                ) as total_days,
                CAST(COUNT(DISTINCT a.date) AS FLOAT) * 100.0 /
                NULLIF((
                    SELECT COUNT(DISTINCT date)
                    FROM attendance_log
                    WHERE YEAR(date) = {y} AND MONTH(date) = {m}
                ), 0) as percentage
            FROM persons p
            LEFT JOIN attendance_log a
                ON p.id = a.person_id
                AND YEAR(a.date) = {y}
                AND MONTH(a.date) = {m}
            GROUP BY p.id, p.name, p.department
            ORDER BY percentage DESC
        """), conn)
    return df

def get_late_arrivals(target_date=None, late_after="09:30"):
    """Returns persons who arrived after late_after time"""
    if target_date is None:
        target_date = date.today()
    d = str(target_date)
    with engine.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT
                a.name,
                p.department,
                MIN(a.timestamp) as arrival_time,
                CASE
                    WHEN CAST(MIN(a.timestamp) AS TIME) > '{late_after}'
                    THEN 'Late'
                    ELSE 'On Time'
                END as status
            FROM attendance_log a
            JOIN persons p ON a.person_id = p.id
            WHERE a.date = '{d}'
            GROUP BY a.person_id, a.name, p.department
            ORDER BY arrival_time
        """), conn)
    return df

def get_unknown_face_count(target_date=None):
    """Returns count of unknown face detections today"""
    if target_date is None:
        target_date = date.today()
    d = str(target_date)
    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT COUNT(*) FROM attendance_log
            WHERE date = '{d}' AND name = 'Unknown'
        """)).scalar()
    return result or 0

def get_weekly_summary():
    """Returns attendance count for last 7 days"""
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT
                date,
                COUNT(DISTINCT person_id) as present_count
            FROM attendance_log
            WHERE date >= CAST(DATEADD(DAY, -7, GETDATE()) AS DATE)
            GROUP BY date
            ORDER BY date
        """), conn)
    return df