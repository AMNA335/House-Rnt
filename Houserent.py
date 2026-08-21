import joblib
import numpy as np
import streamlit as st

# PAGE CONFIG

st.set_page_config(
    page_title="Rent Estimator",
    page_icon="🏠",
    layout="centered",
)


# LOAD MODEL AND ENCODERS


encoders = joblib.load("encoder.pkl")
linear_regression = joblib.load("linear_regression.pkl")


# STYLE — clean, simple, high contrast


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #f7f8fa;
        --card: #ffffff;
        --text: #1c2733;
        --muted: #6b7785;
        --border: #e2e6eb;
        --accent: #8a5a34;
        --accent-soft: #f3e9df;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }

    .stApp {
        background-color: var(--bg);
    }

    /* Header */
    .app-header h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.9rem;
        margin-bottom: 0.15rem;
        color: var(--text);
    }
    .app-header p {
        color: var(--muted);
        font-size: 0.95rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin: 1.4rem 0 0.6rem 0;
    }

    /* Labels */
    label, .stNumberInput label, .stSelectbox label {
        color: var(--text) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }

    /* Number inputs */
    .stNumberInput input {
        background-color: var(--card) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    /* Select boxes (BaseWeb) */
    div[data-baseweb="select"] > div {
        background-color: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }
    div[data-baseweb="select"] span {
        color: var(--text) !important;
    }
    /* Dropdown menu (opened list) */
    ul[data-baseweb="menu"] {
        background-color: var(--card) !important;
    }
    ul[data-baseweb="menu"] li {
        color: var(--text) !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        background-color: var(--accent);
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-top: 1.3rem;
        transition: opacity 0.15s ease;
    }
    .stButton > button:hover {
        opacity: 0.88;
        color: #ffffff;
    }

    /* Result card */
    .result-card {
        background-color: var(--accent-soft);
        border: 1px solid var(--accent);
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        margin-top: 1.5rem;
        text-align: center;
    }
    .result-value {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 2rem;
        color: var(--text);
        margin: 0.1rem 0;
    }
    .result-caption {
        color: var(--muted);
        font-size: 0.85rem;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# HEADER


st.markdown(
    """
    <div class="app-header">
        <h1>🏠 House Rent Estimator</h1>
        <p>Fill in the property details to get an estimated monthly rent.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# NUMERICAL INPUTS


st.markdown('<div class="section-title">Dimensions</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    BHK = st.number_input("BHK", min_value=1, max_value=6, value=2)
with col2:
    Size = st.number_input("Size (sqft)", min_value=10, max_value=8000, value=1000)
with col3:
    Bathroom = st.number_input("Bathroom", min_value=1, max_value=10, value=2)

# ============================================================
# CATEGORICAL INPUTS
# ============================================================

st.markdown('<div class="section-title">Property Details</div>', unsafe_allow_html=True)

col4, col5 = st.columns(2)
with col4:
    Floor = st.selectbox("Floor", encoders["Floor"].classes_)
with col5:
    Area_Type = st.selectbox("Area Type", encoders["Area Type"].classes_)

Area_Locality = st.selectbox("Area Locality", encoders["Area Locality"].classes_)

col6, col7 = st.columns(2)
with col6:
    City = st.selectbox("City", encoders["City"].classes_)
with col7:
    Furnishing_Status = st.selectbox(
        "Furnishing Status", encoders["Furnishing Status"].classes_
    )

st.markdown('<div class="section-title">Tenancy</div>', unsafe_allow_html=True)

col8, col9 = st.columns(2)
with col8:
    Tenant_Preferred = st.selectbox(
        "Tenant Preferred", encoders["Tenant Preferred"].classes_
    )
with col9:
    Point_of_Contact = st.selectbox(
        "Point of Contact", encoders["Point of Contact"].classes_
    )

# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict Rent"):

    # Convert categorical strings into numbers
    Floor_encoded = encoders["Floor"].transform([Floor])[0]
    Area_Type_encoded = encoders["Area Type"].transform([Area_Type])[0]
    Area_Locality_encoded = encoders["Area Locality"].transform([Area_Locality])[0]
    City_encoded = encoders["City"].transform([City])[0]
    Furnishing_Status_encoded = encoders["Furnishing Status"].transform(
        [Furnishing_Status]
    )[0]
    Tenant_Preferred_encoded = encoders["Tenant Preferred"].transform(
        [Tenant_Preferred]
    )[0]
    Point_of_Contact_encoded = encoders["Point of Contact"].transform(
        [Point_of_Contact]
    )[0]

    # CREATE INPUT DATA
    input_data = np.array([[
        BHK,
        Size,
        Floor_encoded,
        Area_Type_encoded,
        Area_Locality_encoded,
        City_encoded,
        Furnishing_Status_encoded,
        Tenant_Preferred_encoded,
        Bathroom,
        Point_of_Contact_encoded
    ]])

    # PREDICTION
    result = linear_regression.predict(input_data)

    # DISPLAY RESULT
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-caption">Estimated Monthly Rent</div>
            <div class="result-value">₹{result[0]:,.0f}</div>
            <div class="result-caption">{BHK} BHK · {Size} sqft · {City}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )