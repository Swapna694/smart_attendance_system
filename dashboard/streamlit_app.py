import sys
sys.path.insert(0, ".")

import streamlit as st
import plotly.express as px
import cv2
import numpy as np
import threading
import time
from datetime import date, datetime
from app.database import (
    get_daily_attendance,
    get_monthly_attendance,
    get_hourly_breakdown,
    get_department_summary,
    get_person_monthly_summary,
    get_total_registered,
    get_absent_today,
    get_attendance_percentage,
    get_late_arrivals,
    get_weekly_summary,
)
from app.detector        import FaceDetector
from app.recognizer      import FaceRecognizer
from app.tracker         import FaceTracker
from app.duplicate_check import DuplicateChecker
from app.database        import log_attendance, load_all_embeddings, save_person_embedding
from app.alerts          import send_attendance_alert

st.set_page_config(page_title="Smart Attendance", layout="wide", page_icon="🎯")

# ── custom style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 4px solid #028090;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #1E2761; }
    .metric-label { font-size: 13px; color: #64748B; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Smart Attendance Dashboard")

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filters")
    selected_date  = st.date_input("Date", value=date.today())
    selected_year  = st.number_input("Year",  value=datetime.now().year, step=1)
    selected_month = st.number_input("Month", value=datetime.now().month, min_value=1, max_value=12)

    st.divider()
    st.subheader("⏰ Late Arrival Setting")
    late_hour   = st.selectbox("Late after hour",   list(range(6, 13)), index=3)
    late_minute = st.selectbox("Late after minute", [0, 15, 30, 45],    index=2)
    late_after  = f"{late_hour:02d}:{late_minute:02d}"
    st.info(f"Late if arrived after **{late_after}**")

    st.divider()
    total         = get_total_registered()
    df_daily      = get_daily_attendance(selected_date)
    df_absent     = get_absent_today(selected_date)
    present_count = len(df_daily)
    absent_count  = len(df_absent)

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total}</div>
        <div class="metric-label">Total Registered</div>
    </div><br>
    <div class="metric-card">
        <div class="metric-value" style="color:#065f46">{present_count}</div>
        <div class="metric-label">Present Today</div>
    </div><br>
    <div class="metric-card">
        <div class="metric-value" style="color:#991b1b">{absent_count}</div>
        <div class="metric-label">Absent Today</div>
    </div>
    """, unsafe_allow_html=True)

# ── tabs ──────────────────────────────────────────────────────────────────────
tab0, tab_reg, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📷 Take Attendance",
    "➕ Register Face",
    "📅 Daily",
    "📊 Monthly Analytics",
    "👤 Person-wise",
    "🏢 Department",
    "⏰ Late Arrivals",
    "📈 Attendance %",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — TAKE ATTENDANCE (live camera)
# ══════════════════════════════════════════════════════════════════════════════
with tab0:
    st.subheader("📷 Live Attendance — Camera")
    st.caption("Click **Start** to open the camera and begin marking attendance automatically.")

    col_btn1, col_btn2 = st.columns(2)
    start_btn = col_btn1.button("▶️ Start Attendance", use_container_width=True)
    stop_btn  = col_btn2.button("⏹️ Stop",             use_container_width=True)

    frame_placeholder  = st.empty()
    status_placeholder = st.empty()
    log_placeholder    = st.empty()

    if "attendance_running" not in st.session_state:
        st.session_state.attendance_running = False
    if "attendance_log_list" not in st.session_state:
        st.session_state.attendance_log_list = []

    if start_btn:
        st.session_state.attendance_running   = True
        st.session_state.attendance_log_list  = []

    if stop_btn:
        st.session_state.attendance_running = False
        status_placeholder.info("⏹️ Attendance stopped.")

    if st.session_state.attendance_running:
        detector   = FaceDetector("models/yolov8n.pt")
        recognizer = FaceRecognizer(threshold=0.9)
        tracker    = FaceTracker()
        dedup      = DuplicateChecker()

        rows = load_all_embeddings()
        recognizer.load_from_db(rows)

        identity_cache = {}
        cap = cv2.VideoCapture(0)

        status_placeholder.success("🟢 Camera running — press Stop to end.")

        from PIL import Image as PILImage

        while st.session_state.attendance_running:
            ret, frame = cap.read()
            if not ret:
                break

            import torch
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(img_rgb)
            boxes, _ = recognizer.mtcnn.detect(pil_img)

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    pid, name = recognizer.identify(face_crop)

                    if pid is not None and not dedup.already_marked(pid):
                        log_attendance(pid, name, 0)
                        dedup.mark(pid)
                        send_attendance_alert(name, department="—")
                        entry = f"✅ {datetime.now().strftime('%H:%M:%S')} — {name} marked present"
                        st.session_state.attendance_log_list.append(entry)

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            # show frame in dashboard
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # show log
            if st.session_state.attendance_log_list:
                log_placeholder.markdown(
                    "**Attendance Log:**\n" +
                    "\n".join(st.session_state.attendance_log_list[-10:])
                )

            time.sleep(0.03)

        cap.release()

# ══════════════════════════════════════════════════════════════════════════════
# TAB REG — REGISTER NEW FACE
# ══════════════════════════════════════════════════════════════════════════════
with tab_reg:
    st.subheader("➕ Register New Person")
    st.caption("Fill in the details below and capture 20 face photos to enroll a new person.")

    col1, col2, col3 = st.columns(3)
    reg_name   = col1.text_input("Full Name",    placeholder="e.g. Sinchana VS")
    reg_emp_id = col2.text_input("Employee ID",  placeholder="e.g. EMP003")
    reg_dept   = col3.text_input("Department",   placeholder="e.g. Engineering")

    col_b1, col_b2 = st.columns(2)
    reg_start = col_b1.button("📸 Start Capturing", use_container_width=True)
    reg_stop  = col_b2.button("💾 Stop & Save",      use_container_width=True)

    reg_frame_placeholder  = st.empty()
    reg_status_placeholder = st.empty()
    reg_progress           = st.empty()

    if "reg_running"    not in st.session_state:
        st.session_state.reg_running    = False
    if "reg_embeddings" not in st.session_state:
        st.session_state.reg_embeddings = []
    if "reg_count"      not in st.session_state:
        st.session_state.reg_count      = 0

    if reg_start:
        if not reg_name or not reg_emp_id or not reg_dept:
            reg_status_placeholder.error("❌ Please fill in Name, Employee ID, and Department first.")
        else:
            st.session_state.reg_running    = True
            st.session_state.reg_embeddings = []
            st.session_state.reg_count      = 0
            reg_status_placeholder.info("📸 Camera open — look at the camera. Capturing automatically...")

    if reg_stop:
        st.session_state.reg_running = False
        if st.session_state.reg_embeddings and reg_name and reg_emp_id and reg_dept:
            import numpy as np_reg
            import pickle
            mean_emb  = np_reg.mean(st.session_state.reg_embeddings, axis=0)
            emb_bytes = pickle.dumps(mean_emb)
            save_person_embedding(reg_name, reg_emp_id, reg_dept, emb_bytes)
            reg_status_placeholder.success(
                f"✅ '{reg_name}' enrolled successfully with "
                f"{st.session_state.reg_count} samples!"
            )
            st.session_state.reg_embeddings = []
            st.session_state.reg_count      = 0
        else:
            reg_status_placeholder.warning("⚠️ No captures yet or details missing.")

    if st.session_state.reg_running:
        from PIL import Image as PILImage2
        detector2   = FaceDetector("models/yolov8n.pt")
        recognizer2 = FaceRecognizer(threshold=0.9)
        cap2        = cv2.VideoCapture(0)

        while st.session_state.reg_running and st.session_state.reg_count < 20:
            ret, frame = cap2.read()
            if not ret:
                break

            img_rgb2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img2 = PILImage2.fromarray(img_rgb2)
            boxes2, _ = recognizer2.mtcnn.detect(pil_img2)

            if boxes2 is not None:
                for box in boxes2:
                    x1, y1, x2, y2 = map(int, box)
                    hh, ww = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(ww, x2), min(hh, y2)
                    face_crop2 = frame[y1:y2, x1:x2]
                    if face_crop2.size == 0:
                        continue
                    emb = recognizer2.get_embedding(face_crop2)
                    if emb is not None:
                        st.session_state.reg_embeddings.append(emb)
                        st.session_state.reg_count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # progress bar
            reg_progress.progress(
                st.session_state.reg_count / 20,
                text=f"Captured {st.session_state.reg_count}/20 samples"
            )

            frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            reg_frame_placeholder.image(frame_disp, channels="RGB", use_column_width=True)
            time.sleep(0.05)

        cap2.release()

        if st.session_state.reg_count >= 20:
            st.session_state.reg_running = False
            import numpy as np_reg2
            import pickle
            mean_emb  = np_reg2.mean(st.session_state.reg_embeddings, axis=0)
            emb_bytes = pickle.dumps(mean_emb)
            save_person_embedding(reg_name, reg_emp_id, reg_dept, emb_bytes)
            reg_status_placeholder.success(
                f"✅ '{reg_name}' enrolled successfully with 20 samples!"
            )
            reg_progress.progress(1.0, text="✅ Enrollment complete!")
            st.session_state.reg_embeddings = []
            st.session_state.reg_count      = 0

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DAILY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Attendance on {selected_date}")
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Present", present_count)
    col2.metric("❌ Absent",  absent_count)
    col3.metric("👥 Total",   total)
    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### ✅ Present Today")
        if df_daily.empty:
            st.info("No one present yet today.")
        else:
            st.dataframe(df_daily, use_container_width=True)
            st.download_button("⬇️ Download Present List", df_daily.to_csv(index=False),
                               f"present_{selected_date}.csv", "text/csv")
    with col_right:
        st.markdown("#### ❌ Absent Today")
        if df_absent.empty:
            st.success("🎉 Everyone is present today!")
        else:
            st.dataframe(df_absent, use_container_width=True)
            st.download_button("⬇️ Download Absent List", df_absent.to_csv(index=False),
                               f"absent_{selected_date}.csv", "text/csv")

    st.divider()
    df_hour = get_hourly_breakdown(selected_date)
    if not df_hour.empty:
        st.markdown("#### 🕐 Arrivals by Hour")
        fig = px.bar(df_hour, x="hour", y="count",
                     labels={"hour": "Hour of Day", "count": "People"},
                     color_discrete_sequence=["#028090"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📆 Last 7 Days Trend")
    df_week = get_weekly_summary()
    if not df_week.empty:
        fig2 = px.line(df_week, x="date", y="present_count", markers=True,
                       labels={"date": "Date", "present_count": "People Present"},
                       color_discrete_sequence=["#1E2761"])
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MONTHLY
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Monthly — {selected_month}/{selected_year}")
    df_mon = get_monthly_attendance(selected_year, selected_month)
    if df_mon.empty:
        st.info("No records for this month.")
    else:
        fig = px.line(df_mon, x="date", y="count", color="name",
                      title="Daily attendance per person")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_mon, use_container_width=True)
        st.download_button("⬇️ Download", df_mon.to_csv(index=False),
                           f"monthly_{selected_year}_{selected_month}.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PERSON-WISE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Person Summary")
    df_person = get_person_monthly_summary(selected_year, selected_month)
    if df_person.empty:
        st.info("No data.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df_person, use_container_width=True)
            st.download_button("⬇️ Download", df_person.to_csv(index=False),
                               f"person_{selected_year}_{selected_month}.csv", "text/csv")
        with col2:
            fig = px.bar(df_person.sort_values("days_present", ascending=False),
                         x="name", y="days_present", color="department",
                         title="Days present per person",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DEPARTMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(f"Department summary — {selected_date}")
    df_dept = get_department_summary(selected_date)
    if df_dept.empty:
        st.info("No records.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(df_dept, names="department", values="present",
                         title="Present by department",
                         color_discrete_sequence=["#028090","#1E2761","#02C39A","#00A896"])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(df_dept, x="department", y="present",
                          title="Count by department",
                          color_discrete_sequence=["#1E2761"])
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(df_dept, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — LATE ARRIVALS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader(f"⏰ Late Arrivals — {selected_date}")
    st.caption(f"Threshold: anyone arriving after **{late_after}** is marked Late")
    df_late = get_late_arrivals(selected_date, late_after)
    if df_late.empty:
        st.info("No attendance records for this date.")
    else:
        late_count   = len(df_late[df_late["status"] == "Late"])
        ontime_count = len(df_late[df_late["status"] == "On Time"])
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 On Time", ontime_count)
        col2.metric("🔴 Late",    late_count)
        col3.metric("📋 Total",   len(df_late))
        st.divider()

        def highlight_status(val):
            if val == "Late":
                return "background-color:#fee2e2;color:#991b1b;font-weight:bold"
            return "background-color:#d1fae5;color:#065f46;font-weight:bold"

        styled = df_late.style.map(highlight_status, subset=["status"])
        st.dataframe(styled, use_container_width=True)
        st.download_button("⬇️ Download", df_late.to_csv(index=False),
                           f"late_{selected_date}.csv", "text/csv")

        late_names = df_late[df_late["status"] == "Late"]["name"].tolist()
        if late_names:
            st.warning(f"🔴 Late today: **{', '.join(late_names)}**")
        else:
            st.success("🎉 Everyone arrived on time today!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ATTENDANCE %
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader(f"Attendance % — {selected_month}/{selected_year}")
    df_pct = get_attendance_percentage(selected_year, selected_month)
    if df_pct.empty:
        st.info("No data for this month.")
    else:
        if "percentage" in df_pct.columns:
            below_75 = len(df_pct[df_pct["percentage"] < 75])
            above_75 = len(df_pct[df_pct["percentage"] >= 75])
            col1, col2 = st.columns(2)
            col1.metric("✅ Above 75%", above_75)
            col2.metric("⚠️ Below 75%", below_75)
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df_pct[["name","department","days_present","total_days","percentage"]],
                         use_container_width=True)
            st.download_button("⬇️ Download", df_pct.to_csv(index=False),
                               f"pct_{selected_year}_{selected_month}.csv", "text/csv")
        with col2:
            fig = px.bar(df_pct.sort_values("percentage", ascending=False),
                         x="name", y="percentage", color="percentage",
                         color_continuous_scale=["#ef4444","#f59e0b","#10b981"],
                         title="Attendance % per person")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            fig.add_hline(y=75, line_dash="dash", line_color="red",
                          annotation_text="75% minimum")
            st.plotly_chart(fig, use_container_width=True)

        if "percentage" in df_pct.columns:
            low = df_pct[df_pct["percentage"] < 75]["name"].tolist()
            if low:
                st.error(f"⚠️ Below 75%: **{', '.join(low)}**")