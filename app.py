from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ChoiceIQ", layout="centered")

BASE = Path(__file__).resolve().parent
DATA = BASE / "Data" / "processed"
MODELS = BASE / "models"

FEATURES = [
    "data_analytics", "programming", "mathematics", "cybersecurity", "networking",
    "problem_solving", "creativity_design", "systems_cloud", "communication_business"
]

FEATURE_LABELS = {
    "data_analytics": "Data and analytics",
    "programming": "Programming",
    "mathematics": "Mathematics",
    "cybersecurity": "Cybersecurity",
    "networking": "Networking",
    "problem_solving": "Problem solving",
    "creativity_design": "Creativity and design",
    "systems_cloud": "Systems and cloud",
    "communication_business": "Communication and business",
}

ANSWER_WEIGHTS = {
    "Data and analytics": {"data_analytics": 1.0, "mathematics": 0.5, "problem_solving": 0.4},
    "Programming": {"programming": 1.0, "problem_solving": 0.4},
    "Cybersecurity": {"cybersecurity": 1.0, "problem_solving": 0.5, "systems_cloud": 0.25},
    "Networking": {"networking": 1.0, "systems_cloud": 0.5, "problem_solving": 0.25},
    "Software development": {"programming": 1.0, "creativity_design": 0.45, "problem_solving": 0.5},

    "Analysing information": {"data_analytics": 1.0, "problem_solving": 0.5, "mathematics": 0.25},
    "Solving technical problems": {"problem_solving": 1.0, "systems_cloud": 0.25},
    "Building applications": {"programming": 1.0, "creativity_design": 0.35},
    "Investigating security problems": {"cybersecurity": 1.0, "problem_solving": 0.75},
    "Working with computer systems": {"systems_cloud": 1.0, "networking": 0.4},

    "Mathematics and logical thinking": {"mathematics": 1.0, "problem_solving": 0.55},
    "Writing code": {"programming": 1.0},
    "Finding and fixing problems": {"problem_solving": 1.0, "systems_cloud": 0.3},
    "Designing digital experiences": {"creativity_design": 1.0, "programming": 0.25},
    "Explaining technical ideas to people": {"communication_business": 1.0, "problem_solving": 0.25},

    "Finding patterns and insights in data": {"data_analytics": 1.0, "mathematics": 0.55},
    "Creating software solutions": {"programming": 1.0, "creativity_design": 0.35},
    "Protecting systems and information": {"cybersecurity": 1.0, "systems_cloud": 0.3},
    "Connecting devices and networks": {"networking": 1.0, "systems_cloud": 0.55},
    "Improving how people use technology": {"creativity_design": 0.85, "communication_business": 0.55},

    "Data and research": {"data_analytics": 1.0, "mathematics": 0.5},
    "Coding and software": {"programming": 1.0, "creativity_design": 0.3},
    "Security and investigation": {"cybersecurity": 1.0, "problem_solving": 0.55},
    "Networks, systems and cloud": {"networking": 0.8, "systems_cloud": 1.0},
    "Technology, business and users": {"communication_business": 1.0, "problem_solving": 0.45, "creativity_design": 0.25},

    "Numbers, trends and evidence": {"data_analytics": 0.9, "mathematics": 0.8},
    "Code and application behaviour": {"programming": 1.0, "problem_solving": 0.5},
    "Threats, risks and vulnerabilities": {"cybersecurity": 1.0, "problem_solving": 0.5},
    "Infrastructure and connectivity": {"networking": 0.85, "systems_cloud": 1.0},
    "People, requirements and processes": {"communication_business": 1.0, "problem_solving": 0.4},

    "A clear logical answer": {"mathematics": 0.7, "problem_solving": 1.0},
    "A working piece of software": {"programming": 1.0, "creativity_design": 0.4},
    "A secure and protected system": {"cybersecurity": 1.0, "systems_cloud": 0.35},
    "A reliable technical environment": {"systems_cloud": 1.0, "networking": 0.6},
    "A solution that works well for users": {"creativity_design": 0.8, "communication_business": 0.7},

    "Working independently on technical tasks": {"problem_solving": 0.8, "programming": 0.45},
    "Collaborating with developers": {"programming": 0.7, "communication_business": 0.55},
    "Investigating issues carefully": {"cybersecurity": 0.55, "problem_solving": 1.0},
    "Supporting users and technical teams": {"communication_business": 1.0, "systems_cloud": 0.45},
    "Planning and improving systems": {"systems_cloud": 0.8, "communication_business": 0.55, "problem_solving": 0.5},

    "Large datasets and reports": {"data_analytics": 1.0, "mathematics": 0.45},
    "Source code and development tools": {"programming": 1.0},
    "Security logs and alerts": {"cybersecurity": 1.0, "data_analytics": 0.35},
    "Servers, devices and networks": {"networking": 0.9, "systems_cloud": 1.0},
    "Designs, requirements and user feedback": {"creativity_design": 0.85, "communication_business": 0.8},

    "Discovering useful insights": {"data_analytics": 1.0, "mathematics": 0.4},
    "Building something from an idea": {"programming": 0.85, "creativity_design": 0.8},
    "Finding weaknesses before they become problems": {"cybersecurity": 0.9, "problem_solving": 0.75},
    "Keeping technology stable and available": {"systems_cloud": 1.0, "networking": 0.7},
    "Helping people make better technology decisions": {"communication_business": 1.0, "problem_solving": 0.55},
}

QUESTIONS = [
    ("1. Which area interests you most?", [
        "Data and analytics", "Programming", "Cybersecurity", "Networking", "Software development"
    ]),
    ("2. What do you enjoy doing most?", [
        "Analysing information", "Solving technical problems", "Building applications",
        "Investigating security problems", "Working with computer systems"
    ]),
    ("3. Which skill sounds most like you?", [
        "Mathematics and logical thinking", "Writing code", "Finding and fixing problems",
        "Designing digital experiences", "Explaining technical ideas to people"
    ]),
    ("4. What type of problem would you rather work on?", [
        "Finding patterns and insights in data", "Creating software solutions",
        "Protecting systems and information", "Connecting devices and networks",
        "Improving how people use technology"
    ]),
    ("5. Which work environment sounds most interesting?", [
        "Data and research", "Coding and software", "Security and investigation",
        "Networks, systems and cloud", "Technology, business and users"
    ]),
    ("6. What information would you prefer to work with?", [
        "Numbers, trends and evidence", "Code and application behaviour",
        "Threats, risks and vulnerabilities", "Infrastructure and connectivity",
        "People, requirements and processes"
    ]),
    ("7. Which outcome would give you the most satisfaction?", [
        "A clear logical answer", "A working piece of software", "A secure and protected system",
        "A reliable technical environment", "A solution that works well for users"
    ]),
    ("8. Which way of working suits you best?", [
        "Working independently on technical tasks", "Collaborating with developers",
        "Investigating issues carefully", "Supporting users and technical teams",
        "Planning and improving systems"
    ]),
    ("9. Which tools or information would you rather spend time with?", [
        "Large datasets and reports", "Source code and development tools", "Security logs and alerts",
        "Servers, devices and networks", "Designs, requirements and user feedback"
    ]),
    ("10. Which statement sounds most motivating to you?", [
        "Discovering useful insights", "Building something from an idea",
        "Finding weaknesses before they become problems", "Keeping technology stable and available",
        "Helping people make better technology decisions"
    ]),
]

@st.cache_data
def load_careers():
    return pd.read_csv(DATA / "choiceiq_careers.csv")

@st.cache_resource
def load_model():
    return joblib.load(MODELS / "cosine_knn.joblib")


def build_user_vector(answers):
    profile = {feature: 0.0 for feature in FEATURES}
    for answer in answers:
        for feature, weight in ANSWER_WEIGHTS.get(answer, {}).items():
            profile[feature] += weight
    vector = np.array([profile[f] for f in FEATURES], dtype=float)
    if vector.max() > 0:
        vector = vector / vector.max()
    return vector


def explain_match(user_vector, row):
    career_vector = row[FEATURES].to_numpy(dtype=float)
    contributions = user_vector * career_vector
    order = np.argsort(contributions)[::-1]
    return [FEATURE_LABELS[FEATURES[i]] for i in order if contributions[i] > 0][:3]


def pretty_name(name):
    return str(name).replace("Ict ", "ICT ").replace("Devops", "DevOps")


def recommend(user_vector, k=3):
    careers = load_careers().copy()
    model = load_model()
    distances, indices = model.kneighbors([user_vector], n_neighbors=k)
    results = careers.iloc[indices[0]].copy().reset_index(drop=True)
    results["similarity"] = 1 - distances[0]
    return results


st.markdown(
    """
    <style>
        .block-container {max-width: 820px; padding-top: 3rem; padding-bottom: 3rem;}
        h1 {font-size: 2.4rem !important; font-weight: 650 !important; letter-spacing: -0.03em;}
        h2, h3 {letter-spacing: -0.02em;}
        [data-testid="stMetricValue"] {font-size: 1.5rem;}
        .stAlert {border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ChoiceIQ")
st.subheader("Find IT careers that match your interests")
st.write(
    "Answer the ten questions below. Your responses are compared with IT career profiles "
    "to identify your three strongest matches."
)

with st.form("career_quiz"):
    answers = []
    for i, (question, options) in enumerate(QUESTIONS):
        answers.append(st.radio(question, options, index=None, key=f"q{i}"))
    submitted = st.form_submit_button("Find my careers", type="primary", use_container_width=True)

if submitted:
    if any(answer is None for answer in answers):
        unanswered = sum(answer is None for answer in answers)
        st.warning(f"Please answer all questions before continuing. {unanswered} question(s) are still unanswered.")
    else:
        user_vector = build_user_vector(answers)
        top = recommend(user_vector, 3)

        st.divider()
        st.header("Your career matches")
        st.write("These are the three careers that most closely match your interests.")

        for i, row in top.iterrows():
            match_percent = max(0, min(100, int(round(float(row["similarity"]) * 100))))
            career = pretty_name(row["career"])
            reasons = explain_match(user_vector, row)

            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.subheader(f"{i + 1}. {career}")
                with c2:
                    st.metric("Match", f"{match_percent}%")

                description = str(row.get("description", ""))
                if description and description.lower() != "nan":
                    st.write(description[:420] + ("..." if len(description) > 420 else ""))

                if reasons:
                    st.write("Why it matches: " + ", ".join(reasons))

                validation = []
                if bool(row.get("sa_high_demand_2024", False)):
                    validation.append("2024 South African High Demand list")
                if bool(row.get("sa_critical_skills_2023", False)):
                    validation.append("2023 South African Critical Skills list")

                if validation:
                    st.write("South Africa validation: " + "; ".join(validation))
                    official_match = str(row.get("sa_official_match", "")).strip()
                    if official_match and official_match.lower() != "nan":
                        st.caption("Closest official occupation: " + official_match)
                else:
                    st.caption("South Africa validation: no direct title match in the two supplied official lists.")

        st.caption(
            "Recommendations are generated from your quiz responses using content-based cosine similarity. "
            "They are intended as career guidance rather than a final career decision."
        )

st.divider()
st.caption("ChoiceIQ | BICT242 Data Scalability and Analytics")
