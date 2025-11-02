import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# CONFIGURATION & LOAD ARTIFACTS
# -------------------------------
st.set_page_config(page_title="Student Exam Score Predictor", layout="centered")

# Paths
ARTIFACTS_DIR = "model_artifacts"

# Load artifacts
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))
    
    best_model = joblib.load(os.path.join(ARTIFACTS_DIR, "best_model.pkl"))
    
    use_poly = os.path.exists(os.path.join(ARTIFACTS_DIR, "poly_features.pkl"))
    poly = joblib.load(os.path.join(ARTIFACTS_DIR, "poly_features.pkl")) if use_poly else None
    
    return scaler, feature_columns, best_model, poly, use_poly

scaler, feature_columns, best_model, poly, use_poly = load_artifacts()

# Title
st.title("🎓 Student Exam Score Predictor")
st.markdown("Predict **Exam Score** based on study habits, environment, and background.")

# -------------------------------
# INPUT FORM
# -------------------------------
with st.form("prediction_form"):
    st.subheader("Enter Student Details")

    col1, col2 = st.columns(2)

    with col1:
        hours_studied = st.slider("Hours Studied", 0, 50, 20)
        attendance = st.slider("Attendance (%)", 0, 100, 85)
        sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
        previous_scores = st.slider("Previous Scores", 0, 100, 75)

    with col2:
        tutoring_sessions = st.slider("Tutoring Sessions", 0, 10, 1)
        physical_activity = st.slider("Physical Activity (hrs/week)", 0, 10, 3)
        motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])

    col3, col4 = st.columns(2)
    with col3:
        access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
        teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
        distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])

    with col4:
        peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        school_type = st.selectbox("School Type", ["Public", "Private"])
        internet_access = st.selectbox("Internet Access", ["No", "Yes"])

    col5, col6 = st.columns(2)
    with col5:
        extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"])
        learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])
        parental_education = st.selectbox(
            "Parental Education Level",
            ["High School", "College", "Postgraduate"]
        )

    submitted = st.form_submit_button("Predict Exam Score")

# -------------------------------
# PREPROCESSING & PREDICTION
# -------------------------------
if submitted:
    # Create input DataFrame
    input_data = {
        'Hours_Studied': hours_studied,
        'Attendance': attendance,
        'Sleep_Hours': sleep_hours,
        'Previous_Scores': previous_scores,
        'Tutoring_Sessions': tutoring_sessions,
        'Physical_Activity': physical_activity,
        'Motivation_Level': motivation_level,
        'Parental_Involvement': parental_involvement,
        'Access_to_Resources': access_to_resources,
        'Family_Income': family_income,
        'Teacher_Quality': teacher_quality,
        'Distance_from_Home': distance_from_home,
        'Peer_Influence': peer_influence,
        'Gender': gender,
        'School_Type': school_type,
        'Internet_Access': internet_access,
        'Extracurricular_Activities': extracurricular,
        'Learning_Disabilities': learning_disabilities,
        'Parental_Education_Level': parental_education
    }

    df_input = pd.DataFrame([input_data])

    # --- Ordinal Encoding ---
    ordinal_map = {
        'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
        'Access_to_Resources':  {'Low': 0, 'Medium': 1, 'High': 2},
        'Family_Income':        {'Low': 0, 'Medium': 1, 'High': 2},
        'Distance_from_Home':   {'Far': 0, 'Moderate': 1, 'Near': 2},
        'Peer_Influence':       {'Negative': -1, 'Neutral': 0, 'Positive': 1},
        'Parental_Education_Level': {'High School': 0, 'College': 1, 'Postgraduate': 2},
        'Teacher_Quality': {'Low': 0, 'Medium': 1, 'High': 2},
        'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2}
    }

    for col, mapping in ordinal_map.items():
        df_input[col] = df_input[col].map(mapping)

    # --- One-Hot Encoding (Nominal) ---
    nominal_cols = ['School_Type', 'Gender', 'Internet_Access', 'Extracurricular_Activities', 'Learning_Disabilities']
    df_input = pd.get_dummies(df_input, columns=nominal_cols, drop_first=True, dtype=int)

    # --- Align columns with training features ---
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_columns]  # Reorder & select only expected columns

    # --- Scale numeric features ---
    num_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours',
                    'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
    df_input[num_features] = scaler.transform(df_input[num_features])

    # --- Polynomial Features (if used) ---
    X_final = poly.transform(df_input) if use_poly else df_input

    # --- Prediction ---
    prediction = best_model.predict(X_final)[0]

    # Clip to realistic range (optional)
    prediction = max(0, min(100, prediction))

    # Display result
    st.success(f"### Predicted Exam Score: **{prediction:.2f}**")

    model_name = "Polynomial (Degree 2)" if use_poly else "Linear Regression"
    st.info(f"Model used: **{model_name}**")

    # Optional: Show input summary
    with st.expander("View Preprocessed Input"):
        st.write(df_input)