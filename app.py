import os

import pandas as pd
import psycopg2
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Titanic Data Analysis",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Titanic Data Analysis")
st.caption("Explore passenger information, survival patterns, and SQL results.")


# ---------------------------------------------------------
# DATABASE CONNECTION
# Uses Streamlit Secrets in the cloud and local
# .streamlit/secrets.toml when running on your computer.
# ---------------------------------------------------------
def get_db_connection():
    try:
        # Preferred method: Streamlit secrets
        db_config = dict(st.secrets["postgres"])
    except Exception:
        # Optional fallback for environment variables
        required = ["PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD"]
        missing = [key for key in required if not os.getenv(key)]
        if missing:
            raise ValueError(
                "Database settings are missing. Add a [postgres] section "
                "to Streamlit Secrets."
            )

        db_config = {
            "host": os.getenv("PGHOST"),
            "database": os.getenv("PGDATABASE"),
            "user": os.getenv("PGUSER"),
            "password": os.getenv("PGPASSWORD"),
            "port": os.getenv("PGPORT", "5432"),
            "sslmode": os.getenv("PGSSLMODE", "require"),
        }

    return psycopg2.connect(
        host=db_config["host"],
        database=db_config["database"],
        user=db_config["user"],
        password=db_config["password"],
        port=db_config.get("port", "5432"),
        sslmode=db_config.get("sslmode", "require"),
        connect_timeout=10,
    )


@st.cache_data(ttl=60)
def load_passenger_data():
    conn = get_db_connection()
    try:
        data = pd.read_sql_query(
            "SELECT * FROM titanic_project.passengers ORDER BY passengerid",
            conn
        )
    finally:
        conn.close()
    return data


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
try:
    df = load_passenger_data()
    st.success("PostgreSQL connected successfully!")

except Exception as error:
    st.error(f"Database connection error: {error}")
    st.info(
        "In Streamlit Cloud, open Manage app → Settings → Secrets "
        "and add your Neon database details under [postgres]."
    )
    st.stop()


# ---------------------------------------------------------
# DASHBOARD OVERVIEW
# ---------------------------------------------------------
st.header("📊 Dataset Overview")

total_passengers = len(df)
survivors = int(df["survived"].sum())
non_survivors = total_passengers - survivors
survival_rate = (survivors / total_passengers * 100) if total_passengers else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Passengers", f"{total_passengers:,}")
col2.metric("Survived", f"{survivors:,}")
col3.metric("Did Not Survive", f"{non_survivors:,}")
col4.metric("Survival Rate", f"{survival_rate:.2f}%")

st.subheader("Passenger Records")
st.dataframe(df.head(10), use_container_width=True)


# ---------------------------------------------------------
# SURVIVAL ANALYSIS
# ---------------------------------------------------------
st.header("📈 Survival Analysis")

left, right = st.columns(2)

with left:
    st.subheader("Survival Count")
    survival_counts = (
        df["survived"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )
    survival_counts.index = ["Did Not Survive", "Survived"]
    st.bar_chart(survival_counts)

with right:
    st.subheader("Passengers by Class")
    class_counts = (
        df["pclass"]
        .value_counts()
        .sort_index()
    )
    class_counts.index = [f"Class {int(x)}" for x in class_counts.index]
    st.bar_chart(class_counts)

st.subheader("Survival Rate by Gender")
gender_survival = (
    df.groupby("sex")["survived"]
    .mean()
    .mul(100)
    .round(2)
    .rename("Survival Rate (%)")
)
st.bar_chart(gender_survival)

st.subheader("Survival Count by Passenger Class")
class_survival = (
    df.groupby(["pclass", "survived"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "Did Not Survive", 1: "Survived"})
)
st.bar_chart(class_survival)

st.subheader("Survival Rate by Class and Gender")
class_gender_rate = (
    df.groupby(["pclass", "sex"])["survived"]
    .mean()
    .mul(100)
    .round(2)
    .reset_index(name="Survival Rate (%)")
)
st.dataframe(class_gender_rate, use_container_width=True)


# ---------------------------------------------------------
# MISSING VALUE ANALYSIS
# ---------------------------------------------------------
st.header("🧹 Missing Value Analysis")

missing_data = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isna().sum().values,
    "Missing Percentage (%)": (
        df.isna().mean().mul(100).round(2).values
    )
})
missing_data = missing_data[missing_data["Missing Values"] > 0]

if missing_data.empty:
    st.success("There are no missing values in the dataset.")
else:
    st.dataframe(missing_data, use_container_width=True)
    st.bar_chart(missing_data.set_index("Column")["Missing Values"])


# ---------------------------------------------------------
# DATA EXPLORER
# Parameterized SQL query
# ---------------------------------------------------------
st.header("🔎 Data Explorer")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    selected_classes = st.multiselect(
        "Passenger Class",
        options=[1, 2, 3],
        default=[1, 2, 3]
    )

with filter_col2:
    selected_genders = st.multiselect(
        "Gender",
        options=["male", "female"],
        default=["male", "female"]
    )

with filter_col3:
    selected_survival = st.multiselect(
        "Survival Status",
        options=[0, 1],
        default=[0, 1],
        format_func=lambda value: "Survived" if value == 1 else "Did Not Survive"
    )

max_records = st.number_input(
    "Maximum records to display",
    min_value=1,
    max_value=891,
    value=100,
    step=10
)

name_search = st.text_input("Search passenger name")

if not selected_classes or not selected_genders or not selected_survival:
    st.warning("Select at least one option in each filter to view records.")
else:
    explorer_sql = """
        SELECT *
        FROM titanic_project.passengers
        WHERE pclass = ANY(%s)
          AND sex = ANY(%s)
          AND survived = ANY(%s)
          AND name ILIKE %s
        ORDER BY passengerid
        LIMIT %s
    """
    explorer_params = (
        selected_classes,
        selected_genders,
        selected_survival,
        f"%{name_search}%",
        int(max_records),
    )

    explorer_conn = get_db_connection()
    try:
        filtered_df = pd.read_sql_query(
            explorer_sql,
            explorer_conn,
            params=explorer_params
        )
    finally:
        explorer_conn.close()

    st.write(f"Records displayed: {len(filtered_df)}")
    st.dataframe(filtered_df, use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=csv_data,
        file_name="titanic_filtered_passengers.csv",
        mime="text/csv"
    )


# ---------------------------------------------------------
# SQL LAB - PREDEFINED READ-ONLY QUERIES
# ---------------------------------------------------------
st.header("💻 SQL Lab")

queries = {
    "Show first 10 passengers": """
        SELECT *
        FROM titanic_project.passengers
        ORDER BY passengerid
        LIMIT 10;
    """,
    "Total passenger count": """
        SELECT COUNT(*) AS total_passengers
        FROM titanic_project.passengers;
    """,
    "Average age": """
        SELECT ROUND(AVG(age)::numeric, 2) AS average_age
        FROM titanic_project.passengers;
    """,
    "Survival rate": """
        SELECT ROUND(100.0 * AVG(survived), 2) AS survival_rate_percentage
        FROM titanic_project.passengers;
    """,
    "Missing values": """
        SELECT
            COUNT(*) FILTER (WHERE age IS NULL) AS missing_age,
            COUNT(*) FILTER (WHERE cabin IS NULL) AS missing_cabin,
            COUNT(*) FILTER (WHERE embarked IS NULL) AS missing_embarked
        FROM titanic_project.passengers;
    """,
    "Fare statistics": """
        SELECT
            ROUND(MIN(fare)::numeric, 2) AS minimum_fare,
            ROUND(AVG(fare)::numeric, 2) AS average_fare,
            ROUND(MAX(fare)::numeric, 2) AS maximum_fare
        FROM titanic_project.passengers;
    """,
    "Passengers by gender": """
        SELECT sex, COUNT(*) AS total_passengers
        FROM titanic_project.passengers
        GROUP BY sex
        ORDER BY sex;
    """,
    "Survival by gender": """
        SELECT
            sex,
            COUNT(*) AS total_passengers,
            SUM(survived) AS survivors,
            ROUND(100.0 * AVG(survived), 2) AS survival_percentage
        FROM titanic_project.passengers
        GROUP BY sex
        ORDER BY sex;
    """,
    "Passengers by class": """
        SELECT pclass, COUNT(*) AS total_passengers
        FROM titanic_project.passengers
        GROUP BY pclass
        ORDER BY pclass;
    """,
    "Average age by class": """
        SELECT
            pclass,
            ROUND(AVG(age)::numeric, 2) AS average_age
        FROM titanic_project.passengers
        GROUP BY pclass
        ORDER BY pclass;
    """,
}

selected_query_name = st.selectbox(
    "Choose an SQL query",
    list(queries.keys())
)

st.code(queries[selected_query_name], language="sql")

if st.button("Run SQL Query"):
    sql_conn = get_db_connection()
    try:
        result_df = pd.read_sql_query(
            queries[selected_query_name],
            sql_conn
        )
        st.success("Query executed successfully!")
        st.dataframe(result_df, use_container_width=True)
    except Exception as error:
        st.error(f"SQL error: {error}")
    finally:
        sql_conn.close()


# ---------------------------------------------------------
# INTERACTIVE TEST
# ---------------------------------------------------------
st.header("📝 Titanic Interactive Test")

questions = [
    {
        "question": "How many passengers are in the dataset?",
        "options": ["500", "891", "1000", "1200"],
        "answer": "891",
        "explanation": "The Titanic training dataset contains 891 passenger records."
    },
    {
        "question": "Which column represents survival?",
        "options": ["Pclass", "Age", "Survived", "Fare"],
        "answer": "Survived",
        "explanation": "The Survived column stores the survival result."
    },
    {
        "question": "What does Survived = 1 mean?",
        "options": ["Died", "Survived", "Missing", "Unknown"],
        "answer": "Survived",
        "explanation": "1 means survived; 0 means did not survive."
    },
    {
        "question": "Which SQL function counts records?",
        "options": ["SUM()", "AVG()", "COUNT()", "MAX()"],
        "answer": "COUNT()",
        "explanation": "COUNT() returns the number of rows or non-NULL values."
    },
    {
        "question": "Which passenger class had the most passengers?",
        "options": ["First", "Second", "Third", "Equal"],
        "answer": "Third",
        "explanation": "Third class had 491 passengers."
    },
    {
        "question": "Which gender had a higher survival rate?",
        "options": ["Male", "Female", "Equal", "Unknown"],
        "answer": "Female",
        "explanation": "Female passengers had a higher survival rate."
    },
    {
        "question": "Which SQL clause groups records?",
        "options": ["WHERE", "GROUP BY", "ORDER BY", "LIMIT"],
        "answer": "GROUP BY",
        "explanation": "GROUP BY combines rows that have matching values."
    },
    {
        "question": "Which column stores passenger age?",
        "options": ["Fare", "Age", "SibSp", "Parch"],
        "answer": "Age",
        "explanation": "The Age column stores each passenger's age."
    },
    {
        "question": "Which SQL function calculates an average?",
        "options": ["COUNT()", "SUM()", "AVG()", "MIN()"],
        "answer": "AVG()",
        "explanation": "AVG() calculates the arithmetic mean."
    },
    {
        "question": "What does NULL represent in SQL?",
        "options": ["Zero", "Empty string", "Missing value", "False"],
        "answer": "Missing value",
        "explanation": "NULL represents an unknown or missing value."
    },
    {
        "question": "Which Python library is used for data analysis?",
        "options": ["Pandas", "Turtle", "Random", "Math"],
        "answer": "Pandas",
        "explanation": "Pandas is used to manipulate and analyse tabular data."
    },
    {
        "question": "Which library helps create this web application?",
        "options": ["NumPy", "Streamlit", "Psycopg2", "Matplotlib"],
        "answer": "Streamlit",
        "explanation": "Streamlit turns Python scripts into interactive web applications."
    },
    {
        "question": "Which library connects Python to PostgreSQL?",
        "options": ["Pandas", "Psycopg2", "Streamlit", "OS"],
        "answer": "Psycopg2",
        "explanation": "Psycopg2 is a PostgreSQL adapter for Python."
    },
    {
        "question": "Which SQL clause filters records?",
        "options": ["WHERE", "GROUP BY", "SELECT", "INSERT"],
        "answer": "WHERE",
        "explanation": "WHERE filters rows based on a condition."
    },
    {
        "question": "How many passengers survived in this dataset?",
        "options": ["549", "342", "891", "177"],
        "answer": "342",
        "explanation": "There are 342 passengers with Survived equal to 1."
    },
]

with st.form("titanic_quiz"):
    user_answers = []

    for index, question in enumerate(questions):
        st.write(f"**{index + 1}. {question['question']}**")
        answer = st.radio(
            "Choose your answer:",
            question["options"],
            key=f"quiz_answer_{index}"
        )
        user_answers.append(answer)

    submitted = st.form_submit_button("Submit Test")

if submitted:
    score = sum(
        user_answers[index] == question["answer"]
        for index, question in enumerate(questions)
    )
    percentage = score / len(questions) * 100

    st.subheader("🎯 Your Test Results")
    result_col1, result_col2 = st.columns(2)
    result_col1.metric("Your Score", f"{score}/{len(questions)}")
    result_col2.metric("Percentage", f"{percentage:.2f}%")

    if percentage >= 80:
        st.success("Excellent work!")
    elif percentage >= 50:
        st.info("Good effort! Keep practising.")
    else:
        st.warning("Keep learning and try again.")

    st.subheader("Answer Explanations")
    for index, question in enumerate(questions):
        if user_answers[index] == question["answer"]:
            st.success(f"Q{index + 1}: Correct")
        else:
            st.error(
                f"Q{index + 1}: Incorrect. "
                f"Correct answer: {question['answer']}"
            )
        st.write(question["explanation"])
