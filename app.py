import sqlite3
from datetime import datetime
import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model pipeline
model = joblib.load("har_model.joblib")

# Initialize SQLite database
os.makedirs("data", exist_ok=True)

conn = sqlite3.connect("data/predictions.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    activity TEXT,
    confidence REAL
)
""")

conn.commit()
conn.close()

# Activity labels
activity_labels = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

# Expected feature names from the trained model
expected_features = list(model.feature_names_in_)

# Page configuration
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide"
)

st.title("Human Activity Recognition System")

st.write(
    "Upload accelerometer and gyroscope feature measurements "
    "to predict human activities."
)

st.divider()

# CSV file upload
uploaded_file = st.file_uploader(
    "Upload sensor feature CSV",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    # Validate duplicate column names
    if data.columns.duplicated().any():
        st.error("CSV contains duplicate column names.")

    else:

        # Check for missing required features
        missing_features = [
            feature for feature in expected_features
            if feature not in data.columns
        ]

        if missing_features:
            st.error(
                "Missing required features: "
                + ", ".join(missing_features)
            )

        elif data.empty:
            st.error("The uploaded CSV contains no data.")

        else:

            # Select features in the exact training order
            input_data = data[expected_features].copy()

            # Convert values to numeric
            input_data = input_data.apply(
                pd.to_numeric, errors="coerce"
            )

            # Check missing or invalid values
            if input_data.isnull().values.any():

                st.error(
                    "The uploaded data contains missing "
                    "or non-numeric feature values."
                )

            else:

                st.success("CSV validation successful!")

                if st.button("Predict Activity"):

                    # Generate predictions
                    predictions = model.predict(input_data)

                    # Generate prediction probabilities
                    probabilities = model.predict_proba(input_data)

                    # Get confidence of the predicted class
                    confidence = probabilities.max(axis=1) * 100

                    # Create results DataFrame
                    results = pd.DataFrame({
                        "Predicted Activity": [
                            activity_labels[int(label)]
                            for label in predictions
                        ],
                        "Confidence (%)": confidence.round(2)
                    })

                    st.divider()

                    st.subheader("Activity Statistics")

                    # Count predictions for each activity
                    activity_counts = results["Predicted Activity"].value_counts()

                    # Display activity counts
                    st.bar_chart(activity_counts)

                    # Display total predictions
                    st.metric(
                        "Total Predictions",
                        len(results)
                    )

                    # Display most frequently predicted activity
                    st.metric(
                        "Most Common Activity",
                        activity_counts.idxmax()
                    )

                                        # Store predictions in SQLite
                    conn = sqlite3.connect("data/predictions.db")

                    cursor = conn.cursor()

                    for index, row in results.iterrows():

                        cursor.execute("""
                        INSERT INTO prediction_history
                        (timestamp, activity, confidence)
                        VALUES (?, ?, ?)
                        """, (
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            row["Predicted Activity"],
                            row["Confidence (%)"]
                        ))

                    conn.commit()
                    conn.close()

                    st.success("Prediction history saved successfully!")

                    st.subheader("Prediction Results")

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    st.success(
                        "Successfully predicted "
                        + str(len(results))
                        + " activity records."
                    )

                st.divider()

                st.subheader("Prediction History")

                conn = sqlite3.connect("data/predictions.db")

                history = pd.read_sql_query(
                    "SELECT * FROM prediction_history ORDER BY id DESC",
                    conn
                )

                conn.close()

                if not history.empty:

                    st.dataframe(
                        history,
                        use_container_width=True
                    )

                else:

                    st.info("No prediction history available.")
