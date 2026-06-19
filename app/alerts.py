import yagmail
from datetime import datetime

SENDER_EMAIL    = "sinchanavs04@gmail.com"
SENDER_PASSWORD = "kvscknmsvpokccro"
ALERT_RECIPIENT = "swapnavs694@gmail.com"

def send_attendance_alert(name: str, department: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        yag = yagmail.SMTP(SENDER_EMAIL, SENDER_PASSWORD)
        yag.send(
            to      = ALERT_RECIPIENT,
            subject = f"Attendance: {name} - {now}",
            contents= f"Name: {name}\nDepartment: {department}\nTime: {now}"
        )
        print(f"[Alert] Email sent for {name}")
    except Exception as e:
        print(f"[Alert] Failed: {e}")



        