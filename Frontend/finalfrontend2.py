import streamlit as st

USER_CREDENTIALS = {
    "admin": "admin123",
    "teacher": "teach2025"
}


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""


if not st.session_state["authenticated"]:
    st.set_page_config(page_title="🔐 Login - Smart Classroom", page_icon="📚", layout="centered")
    st.title("🔐 Smart Classroom Login")

    with st.form("login_form"):
        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")
        login_button = st.form_submit_button("🔓 Login")

    if login_button:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success("✅ Login successful!")
            st.rerun()

        else:
            st.error("❌ Invalid username or password")

    st.stop()


import streamlit as st
import pandas as pd
from datetime import datetime
import os
import requests
import plotly.express as px

API_URL = "http://localhost:8000"



st.set_page_config(page_title="🎓 Smart Classroom Management System by FUTURE MINDS", layout="wide", page_icon="📚")


def initialize_csv(file, columns):
    if not os.path.exists(file):
        pd.DataFrame(columns=columns).to_csv(file, index=False)

initialize_csv("schedules.csv", ["Subject", "Topic", "Date", "Time"])
initialize_csv("attendance.csv", ["Class", "Student", "Date"])
initialize_csv("performance.csv", ["Student", "Subject", "Marks", "Engagement Score"])

def fix_performance_csv():
    expected_cols = ["Student", "Subject", "Marks", "Engagement Score"]
    try:
        df = pd.read_csv("performance.csv", on_bad_lines='skip')
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None
        df = df[expected_cols]
        df.to_csv("performance.csv", index=False)
    except Exception as e:
        st.error(f"Error fixing performance.csv: {e}")


fix_performance_csv()


st.markdown(""" 
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: None;
        padding: 0.5em 1em;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)


st.sidebar.title("📚 Smart Classroom Menu")
st.sidebar.markdown(f"👋 Welcome, **{st.session_state['username'].capitalize()}**")
if st.sidebar.button("🚪 Logout"):
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""
    st.experimental_rerun()

page = st.sidebar.radio("✨ Navigate to", ["🏠 Home", "📅 Schedule Class", "✅ Attendance", "📊 Student Performance"])

student_name = st.sidebar.text_input("👤 Enter Student Name to Register")
if st.sidebar.button("📝 Register Student"):
    if student_name:
        try:
            res = requests.post(f"{API_URL}/register", json={"name": student_name})
            if res.status_code == 200:
                st.success(res.json()["status"])
            else:
                st.error(f"Error: {res.json().get('detail', 'Unknown')}")
        except Exception as e:
            st.error(f"Failed to register: {e}")
    else:
        st.warning("Please enter a name.")


st.sidebar.markdown("---")
run_session = st.sidebar.button("📷 Start Smart Classroom Session")

show_attendance = st.sidebar.button("👁 Show Latest Attendance")

if show_attendance:
    st.subheader("🧑‍🎓 Students Marked Present in Last Session")
    try:
        res = requests.get(f"{API_URL}/get-latest-attendance")
        if res.status_code == 200:
            data = res.json()
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df.style.set_properties(**{'background-color': '#e8f5e9'}))
            else:
                st.warning("⚠️ No students marked present yet.")
        else:
            st.warning("❌ Failed to load latest attendance log.")
    except Exception as e:
        st.error(f"Error: {e}")


API_URL = "http://localhost:8000"

if run_session:
    st.info("Launching Smart Classroom Session...")
    with st.spinner("Please wait while the session runs..."):
        try:
            res = requests.post(f"{API_URL}/start-session")
            if res.status_code == 200:
                st.success("✅ Session launched! Press 'q' in the webcam window to stop.")
            else:
                st.error(f"❌ Failed to start session: {res.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error calling backend: {e}")


if page == "🏠 Home":
    st.title(" Smart Class Management System")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📅 Classes Scheduled", len(pd.read_csv("schedules.csv")))
    with col2:
        st.metric("✅ Attendance Entries", len(pd.read_csv("attendance.csv")))
    with col3:
        try:
            perf_df = pd.read_csv("performance.csv", on_bad_lines='skip')
            expected_cols = ["Student", "Subject", "Marks", "Engagement Score"]
            for col in expected_cols:
                if col not in perf_df.columns:
                    perf_df[col] = None
            st.metric("📊 Performance Records", len(perf_df))
        except Exception as e:
            st.error(f"Error loading performance records: {e}")
            st.metric("📊 Performance Records", 0)

    st.markdown("###  Features Overview")
    st.success(""" 
    - 📅 *Schedule Classes* with date and time  
    - ✅ *Record Attendance* for each class  
    - 📊 *Track Performance & Engagement* of each student
    """)

    st.markdown("---")
    st.markdown("###  Why Smart Class?")
    st.info(""" 
    - Empower teachers with *easy-to-use tools*  
    - Boost student engagement & tracking  
    - Make education *smarter* and *data-driven*  
    """)


elif page == "📅 Schedule Class":
    st.title("📅 Schedule a New Class")
    st.markdown("Use the form below to create a class schedule.")

    with st.form("schedule_form"):
        col1, col2 = st.columns(2)
        with col1:
            subject = st.text_input("📘 Subject")
        with col2:
            topic = st.text_input("📝 Topic")

        col3, col4 = st.columns(2)
        with col3:
            date = st.date_input("📆 Date")
        with col4:
            time = st.time_input("⏰ Time", datetime.now().time())

        submitted = st.form_submit_button("✅ Schedule Class")

    if submitted:
        df = pd.DataFrame([[subject, topic, date, time]], columns=["Subject", "Topic", "Date", "Time"])
        df.to_csv("schedules.csv", mode='a', header=False, index=False)
        st.success(f"🎉 Class scheduled for *{subject} - {topic}* on *{date} at {time.strftime('%I:%M %p')}*")

    st.markdown("---")
    st.markdown("### 📚 Existing Class Schedules")
    schedule_df = pd.read_csv("schedules.csv")
    ##st.dataframe(schedule_df.style.set_properties({'background-color': '#e8f0fe', 'color': 'black'}))
    st.dataframe(schedule_df.style.set_properties(subset=schedule_df.columns,**{'background-color': '#e8f0fe', 'color': 'black'}))



elif page == "✅ Attendance":
    st.title("✅ Mark Attendance")
    st.markdown("Use the interface below to record student attendance.")

    with st.expander("📌 Select Class and Mark Attendance", expanded=True):
        class_name = st.selectbox("🏷 Choose Class", ["CS-I", "CS-II", "CS-DS-I", "CS-DS-II"])
        students = ["Harsh Vardhan", "Harshdeep", "Sneha", "Ishvar","Kartikeya","Utkarsh"]
        present_students = st.multiselect("👥 Students Present", students)

        attendance_date = st.date_input("📆 Select Date", datetime.now().date())

        if st.button("📤 Submit Attendance"):
            data = [[class_name, student, attendance_date] for student in present_students]
            df = pd.DataFrame(data, columns=["Class", "Student", "Date"])
            df.to_csv("attendance.csv", mode='a', header=False, index=False)
            st.success(f"📍 Attendance marked for *{class_name}* on {attendance_date}")
            st.markdown(f"*Present Students:* {', '.join(present_students)}")


elif page == "📊 Student Performance":
    st.title("📊 Student Performance & Engagement Tracker")
    st.markdown("Record performance and engagement data for each student.")

    with st.form("performance_form"):
        student = st.selectbox("🎓 Select Student", ["Harsh Vardhan", "Harshdeep", "Sneha", "Ishvar","Kartikeya","Utkarsh"])
        subject = st.text_input("📘 Subject")
        marks = st.slider("📈 Marks (out of 100)", 0, 100, 75)
        engagement = st.slider("Engagement Score (out of 100)", 0, 100, 50)

        st.markdown("#### Engagement Observations")
        col1, col2, col3 = st.columns(3)
        with col1:
            yawning = st.checkbox("🥱 Yawning")
        with col2:
            blinking = st.checkbox("👁 Blinking Fast")
        with col3:
            looking_away = st.checkbox("👀 Looking Away")
        col4, col5, col6 = st.columns(3)
        with col4:
            confused = st.checkbox("😕 Confused")
        with col5:
            stressed = st.checkbox("😫 Stressed")
        with col6:
            eyes_closed = st.checkbox("😴 Eyes Closed")
        col7 = st.columns(1)[0]
        with col7:
            talking = st.checkbox("🗣 Talking")

        submit_perf = st.form_submit_button("✅ Submit Performance")

    if submit_perf:
        if not (0 <= marks <= 100):
            st.error("Marks must be between 0 and 100.")
        elif not (0 <= engagement <= 100):
            st.error("Engagement score must be between 0 and 100.")
        else:
            df = pd.DataFrame([[student, subject, marks, engagement]], columns=["Student", "Subject", "Marks", "Engagement Score"])
            df.to_csv("performance.csv", mode='a', header=False, index=False)
            st.success(f"📝 Recorded *{marks} marks* and engagement score *{engagement}/10* in *{subject}* for *{student}*.")

            if yawning or blinking or looking_away:
                st.warning("⚠ Engagement Alert: Signs of distraction detected!")


    st.markdown("---")
    st.markdown("### 📈 Student Performance Records")

    try:
        perf_df = pd.read_csv("performance.csv", on_bad_lines='skip')
        expected_cols = ["Student", "Subject", "Marks", "Engagement Score"]
        for col in expected_cols:
            if col not in perf_df.columns:
                perf_df[col] = None

        st.dataframe(perf_df.style.background_gradient(cmap='Greens'))


        st.markdown("### 🌟 Enhanced Engagement Score Chart")


        perf_df["Engagement Score"] = pd.to_numeric(perf_df["Engagement Score"], errors='coerce')
        perf_df["Marks"] = pd.to_numeric(perf_df["Marks"], errors='coerce')
        perf_df = perf_df.dropna(subset=["Engagement Score"])

        if perf_df.empty:
            st.warning("⚠ No valid engagement data available.")
        else:
            fig = px.bar(
                perf_df,
                x="Student",
                y="Engagement Score",
                color="Subject",
                barmode="group",
                text_auto=".2s",
                opacity=0.8,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )


            fig.update_traces(
                textfont_size=14,
                textangle=0,
                textposition="outside",
                cliponaxis=False,
                marker_line_width=1.5,
                marker_line_color="black",
                textfont_color="red"
            )

            fig.update_layout(
                xaxis_title="👩‍🎓 Student",
                yaxis_title=" Engagement Score (Out of 100)",
                title="📊 Student Engagement Scores by Subject",
                title_x=0.5,
                height=500,
                plot_bgcolor="#f9f9f9",
                paper_bgcolor="#f9f9f9",
                font=dict(size=16),
                bargap=0.2,
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating engagement chart: {e}")