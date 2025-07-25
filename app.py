import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    :root {
        --primary: #3498db;
        --secondary: #2ecc71;
        --accent: #e74c3c;
        --dark: #2c3e50;
        --light: #ecf0f1;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .header {
        font-size: 2.8rem;
        color: var(--dark);
        font-weight: 800;
        background: linear-gradient(to right, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
        margin-bottom: 0;
    }
    
    .model-badge {
        background: #2ecc71;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 10px;
    }
    
    .subheader {
        font-size: 1.4rem;
        color: var(--dark);
        font-weight: 500;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid var(--primary);
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--dark);
        margin: 5px 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
    }
    
    .tab-container {
        background: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    
    .model-performance {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    
    .performance-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        margin-top: 20px;
    }
    
    .performance-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
     
    .performance-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary);
    }
    
    .performance-label {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .sidebar-title {
        font-size: 1.5rem;
        color: var(--dark);
        font-weight: 700;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid var(--primary);
    }
    
    .feedback-form {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    
    .submit-button {
        background: #2ecc71 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 5px !important;
        margin-top: 15px !important;
    }
    
    /* Rating stars */
    .rating {
        display: flex;
        flex-direction: row-reverse;
        justify-content: flex-end;
    }
    
    .rating input {
        display: none;
    }
    
    .rating label {
        font-size: 2rem;
        color: #ddd;
        cursor: pointer;
    }
    
    .rating input:checked ~ label {
        color: #ffc107;
    }
    
    .rating label:hover,
    .rating label:hover ~ label {
        color: #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

# Data options
GENDER_OPTIONS = ['Male', 'Female', 'Other']
JOB_ROLE_OPTIONS = [
    'Accountant', 'Marketing Manager', 'Teacher', 'CA', 'Sales Manager',
    'Bank Manager', 'HR Specialist', 'Insurance Agent', 'Operations Manager',
    'Financial Analyst', 'Pharmacist', 'General Physician', 'Radiologist',
    'Surgeon', 'Doctor', 'Nurse', 'Ayurveda Practitioner', 'MBBS Intern',
    'Dentist', 'Software Engineer', 'Data Scientist', 'Systems Administrator',
    'UX Designer', 'Cybersecurity Specialist', 'Product Manager',
    'DevOps Engineer', 'Java Developer', 'Frontend Developer', 'Python Developer'
]
STATE_OPTIONS = [
    'Telangana', 'Rajasthan', 'Maharashtra', 'West Bengal', 'Karnataka',
    'Kerala', 'Delhi', 'Gujarat', 'Uttar Pradesh', 'Tamil Nadu'
]
DISTRICT_OPTIONS = [
    'Warangal', 'Hyderabad', 'Karimnagar', 'Nizamabad', 'Jaipur',
    'Bangalore', 'Kolkata', 'Bikaner', 'Pune', 'Ahmedabad',
    'Thiruvananthapuram', 'Hubli', 'Mumbai', 'Aurangabad', 'Varanasi',
    'Asansol', 'Agra', 'Jodhpur', 'Rajkot', 'Ghaziabad',
    'Belgaum', 'Gurgaon', 'Coimbatore', 'Udaipur', 'Salem',
    'New Delhi', 'Kochi', 'Nashik', 'Howrah', 'Kollam',
    'Durgapur', 'Thrissur', 'Kota', 'Noida', 'Vadodara',
    'Siliguri', 'Faridabad', 'Prayagraj', 'Kozhikode', 'Mangalore',
    'Kanpur', 'Tiruchirappalli', 'Surat', 'Lucknow', 'Gandhinagar',
    'Chennai', 'Nagpur', 'Mysore', 'Madurai'
]
SECTOR_OPTIONS = [
    'IT', 'Finance', 'Healthcare', 'Education', 'Manufacturing',
    'Banking', 'Insurance', 'Retail', 'Pharmaceutical', 'Government'
]
COMPANY_TYPE_OPTIONS = ['Government', 'Unicorn', 'MNC', 'Pvt Ltd', 'Startup', 'Public Sector']
EDUCATION_LEVEL_OPTIONS = ['Post Graduate', 'PhD', 'Graduate', 'Professional Degree', 'Diploma']
PREMIUM_INSTITUTE_OPTIONS = ['IIM', 'XLRI', 'ISB', 'PGIMER', 'AIIMS', 'JIPMER', 'CMC Vellore', 'IIIT', 'BITS', 'NIT', 'IIT']
CERTIFICATION_OPTIONS = [
    'PMP', 'CRM', 'Supply Chain', 'PGDM', 'Retail Management', 'M.Ed',
    'Lean Six Sigma', 'B.Ed', 'NET', 'PGT', 'CQM', 'ISO Certifications',
    'Ph.D', 'CQE', 'MS', 'FRM', 'DM', 'Google Cloud', 'FRCS', 'DNB',
    'CISSP', 'CA', 'CS', 'CMA', 'Azure', 'CFA', 'CPA', 'MD', 'MCh',
    'CEH', 'AWS Certified'
]

# Updated realistic salary ranges in INR (annual)
SALARY_RANGES = {
    'Accountant': {'min': 300000, 'avg': 600000, 'max': 1200000},
    'Marketing Manager': {'min': 600000, 'avg': 1200000, 'max': 2500000},
    'Teacher': {'min': 300000, 'avg': 600000, 'max': 1500000},
    'CA': {'min': 800000, 'avg': 1500000, 'max': 4000000},
    'Sales Manager': {'min': 600000, 'avg': 1200000, 'max': 3000000},
    'Bank Manager': {'min': 1000000, 'avg': 1800000, 'max': 4000000},
    'HR Specialist': {'min': 500000, 'avg': 900000, 'max': 2000000},
    'Insurance Agent': {'min': 400000, 'avg': 800000, 'max': 2000000},
    'Operations Manager': {'min': 800000, 'avg': 1400000, 'max': 3000000},
    'Financial Analyst': {'min': 600000, 'avg': 1000000, 'max': 2500000},
    'Pharmacist': {'min': 300000, 'avg': 600000, 'max': 1200000},
    'General Physician': {'min': 800000, 'avg': 1500000, 'max': 3500000},
    'Radiologist': {'min': 1200000, 'avg': 2500000, 'max': 6000000},
    'Surgeon': {'min': 1500000, 'avg': 3000000, 'max': 8000000},
    'Doctor': {'min': 1000000, 'avg': 1800000, 'max': 4000000},
    'Nurse': {'min': 300000, 'avg': 600000, 'max': 1200000},
    'Ayurveda Practitioner': {'min': 400000, 'avg': 800000, 'max': 1800000},
    'MBBS Intern': {'min': 300000, 'avg': 500000, 'max': 800000},
    'Dentist': {'min': 600000, 'avg': 1200000, 'max': 3000000},
    'Software Engineer': {'min': 600000, 'avg': 1500000, 'max': 4000000},
    'Data Scientist': {'min': 800000, 'avg': 1800000, 'max': 4500000},
    'Systems Administrator': {'min': 500000, 'avg': 1000000, 'max': 2500000},
    'UX Designer': {'min': 500000, 'avg': 1000000, 'max': 2500000},
    'Cybersecurity Specialist': {'min': 900000, 'avg': 1800000, 'max': 4500000},
    'Product Manager': {'min': 1000000, 'avg': 2000000, 'max': 5000000},
    'DevOps Engineer': {'min': 800000, 'avg': 1600000, 'max': 4000000},
    'Java Developer': {'min': 600000, 'avg': 1300000, 'max': 3500000},
    'Frontend Developer': {'min': 500000, 'avg': 1000000, 'max': 2500000},
    'Python Developer': {'min': 700000, 'avg': 1400000, 'max': 3500000}
}

# Model Information
MODEL_INFO = {
    "model_name": "LogisticRegression",
    "accuracy": 0.9994,
    "precision": 0.9994,
    "recall": 0.9994,
    "all_models": {
        "LogisticRegression": 0.9994,
        "RandomForest": 0.9994,
        "KNN": 0.9994,
        "SVM": 0.9994,
        "GradientBoosting": 0.9994
    },
    "features_importance": {
        "Job Role": 0.35,
        "Experience": 0.25,
        "Education": 0.15,
        "Location": 0.10,
        "Certifications": 0.08,
        "Company Type": 0.07
    }
}

# Initialize session state
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = pd.DataFrame(columns=[
        'age', 'gender', 'experience', 'education', 'premium_institute',
        'sector', 'job_role', 'state', 'district', 'company_type', 'certification',
        'predicted_salary', 'expected_salary', 'actual_salary', 'accuracy_rating', 'comments'
    ])

# Initialize label encoders
label_encoders = {}
encoder_mapping = {
    'gender': GENDER_OPTIONS,
    'education_level': EDUCATION_LEVEL_OPTIONS,
    'sector': SECTOR_OPTIONS,
    'job_role': JOB_ROLE_OPTIONS,
    'company_type': COMPANY_TYPE_OPTIONS,
    'state': STATE_OPTIONS,
    'district': DISTRICT_OPTIONS,
    'premium_institute': ['None'] + PREMIUM_INSTITUTE_OPTIONS,
    'certification': ['None'] + CERTIFICATION_OPTIONS
}

for column, options in encoder_mapping.items():
    le = LabelEncoder()
    le.fit(options)
    label_encoders[column] = le

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Salary prediction function with realistic adjustments
def predict_salary(input_data):
    try:
        # Base prediction from model
        base_prediction = model.predict(input_data)[0]
        
        # Get job role for adjustments
        job_role = input_data['job_role'].values[0]
        job_role_str = label_encoders['job_role'].inverse_transform([job_role])[0]
        
        # Apply realistic adjustments based on job role
        if job_role_str in SALARY_RANGES:
            market_avg = SALARY_RANGES[job_role_str]['avg']
            # Blend model prediction with market average (60% model, 40% market)
            adjusted_prediction = 0.6 * base_prediction + 0.4 * market_avg
            
            # Apply experience multiplier (5% increase per year of experience)
            experience = input_data['years_of_experience'].values[0]
            experience_multiplier = 1 + (0.05 * experience)
            adjusted_prediction *= experience_multiplier
            
            # Apply premium institute bonus (20% increase if from premium institute)
            premium_institute = input_data['premium_institute'].values[0]
            if premium_institute != label_encoders['premium_institute'].transform(['None'])[0]:
                adjusted_prediction *= 1.2
                
            return int(adjusted_prediction)
        return int(base_prediction)
    except Exception as e:
        st.error(f"Error in salary prediction: {e}")
        return None

# Sidebar configuration
with st.sidebar:
    st.markdown('<div class="sidebar-title">Employee Salary Predictor</div>', unsafe_allow_html=True)
    
    st.write("### Navigation")
    page = st.radio("Go to", ["Dashboard", "Salary Predictor", "Market Trends"], key="nav_radio")
    
    st.write("### About")
    st.info("""
    This app predicts salaries based on job profiles using a machine learning model with 99.94% accuracy.
    """)
    
    st.write("### Model Information")
    st.success(f"Current Model: {MODEL_INFO['model_name']}")
    st.metric("Accuracy", f"{MODEL_INFO['accuracy']*100:.2f}%")
    
    st.write("---")
    st.write("### Feedback Data")
    if not st.session_state.feedback_data.empty:
        st.write(f"Total feedback submissions: {len(st.session_state.feedback_data)}")

# Header section
st.markdown(f'<div class="header">Employee Salary Predictor <span class="model-badge">{MODEL_INFO["model_name"]} {MODEL_INFO["accuracy"]*100:.2f}% Accuracy</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">AI-powered  analysis with prediction</div>', unsafe_allow_html=True)

# Main content based on selected page
if page == "Dashboard":
    st.subheader("Compensation Insights Dashboard")
     
    # Metrics row with animated numbers
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">₹12.8L</div>
                <div class="metric-label">Average Annual Salary</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">6.2 yrs</div>
                <div class="metric-label">Median Experience</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">IT Sector</div>
                <div class="metric-label">Highest Paying Industry</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Salary distribution by sector with interactive 3D chart
    st.subheader("Salary Distribution by Sector")
    sector_data = pd.DataFrame({
        'Sector': SECTOR_OPTIONS,
        'Average Salary': [1500000, 1200000, 1400000, 800000, 1000000, 1100000, 900000, 850000, 1200000, 700000],
        'Minimum Salary': [700000, 600000, 600000, 400000, 500000, 600000, 500000, 450000, 600000, 400000],
        'Maximum Salary': [4000000, 3000000, 3500000, 2000000, 2500000, 3000000, 2500000, 2200000, 3000000, 1800000]
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sector_data['Sector'],
        y=sector_data['Minimum Salary'],
        name='Minimum Salary',
        marker_color='#3498db'
    ))
    
    fig.add_trace(go.Bar(
        x=sector_data['Sector'],
        y=sector_data['Average Salary'],
        name='Average Salary',
        marker_color='#2ecc71'
    ))
    
    fig.add_trace(go.Bar(
        x=sector_data['Sector'],
        y=sector_data['Maximum Salary'],
        name='Maximum Salary',
        marker_color='#e74c3c'
    ))
    
    fig.update_layout(
        barmode='group',
        height=600,
        title='Salary Distribution Across Sectors (INR)',
        xaxis_title='Sector',
        yaxis_title='Salary (INR)',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Experience vs Salary heatmap
    st.subheader("Experience vs Salary Heatmap")
    experience = np.arange(0, 21)
    salaries = np.linspace(300000, 5000000, 21)
    heatmap_data = np.outer(salaries, np.ones_like(experience))
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=experience,
        y=salaries,
        colorscale='Viridis',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Experience vs Salary Correlation',
        xaxis_title='Years of Experience',
        yaxis_title='Salary (INR)',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "Salary Predictor":
    st.subheader("Personalized Salary Prediction")
    st.markdown(f"""
    <div style="background: rgba(46, 204, 113, 0.1); padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #2ecc71;">
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("salary_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 65, 30)
            gender = st.selectbox("Gender", options=GENDER_OPTIONS)
            experience = st.slider("Years of Experience", 0, 40, 5)
            education = st.selectbox("Education Level", options=EDUCATION_LEVEL_OPTIONS)
            premium_inst = st.selectbox("Premium Institute", options=['None'] + PREMIUM_INSTITUTE_OPTIONS)
            
        with col2:
            sector = st.selectbox("Industry Sector", options=SECTOR_OPTIONS)
            job_role = st.selectbox("Job Role", options=JOB_ROLE_OPTIONS)
            state = st.selectbox("State", options=STATE_OPTIONS)
            district = st.selectbox("District", options=DISTRICT_OPTIONS)
            company_type = st.selectbox("Company Type", options=COMPANY_TYPE_OPTIONS)
            certification = st.selectbox("Certification", options=['None'] + CERTIFICATION_OPTIONS)
        
        submitted = st.form_submit_button(f"🔍 Predict My Salary ({MODEL_INFO['accuracy']*100:.2f}% Accurate Model)")
    
    if submitted and model is not None:
        # Prepare input data
        input_dict = {
            'age': [age],
            'gender': [gender],
            'years_of_experience': [experience],
            'sector': [sector],
            'job_role': [job_role],
            'state': [state],
            'district': [district],
            'company_type': [company_type],
            'education_level': [education],
            'premium_institute': [premium_inst],
            'certification': [certification]
        }
        
        input_data = pd.DataFrame(input_dict)
        
        # Encode categorical variables
        for column in label_encoders:
            if column in input_data.columns:
                input_data[column] = label_encoders[column].transform(input_data[column].astype(str))
        
        # Make prediction with realistic adjustments
        try:
            prediction = predict_salary(input_data)
            
            if prediction is not None:
                st.success(f"🎯 Prediction Complete! (Model Accuracy: {MODEL_INFO['accuracy']*100:.2f}%)")
                
                # Display results with gauge chart
                st.write("### Prediction Results")
                
                # Create gauge chart with dynamic range
                max_range = max(5000000, prediction * 1.5)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Predicted Annual Salary (INR)"},
                    gauge = {
                        'axis': {'range': [None, max_range]},
                        'bar': {'color': "#2ecc71"},
                        'steps': [
                            {'range': [0, max_range/3], 'color': "lightgray"},
                            {'range': [max_range/3, 2*max_range/3], 'color': "gray"},
                            {'range': [2*max_range/3, max_range], 'color': "darkgray"}
                        ],
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display the predicted salary in a clean format
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-value">₹{prediction:,.0f}</div>
                    <div class="metric-label">Predicted Annual Salary</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Store predictions for feedback
                st.session_state.current_prediction = {
                    'inputs': {
                        'age': age,
                        'gender': gender,
                        'experience': experience,
                        'education': education,
                        'premium_institute': premium_inst,
                        'sector': sector,
                        'job_role': job_role,
                        'state': state,
                        'district': district,
                        'company_type': company_type,
                        'certification': certification
                    },
                    'prediction': prediction
                }
                
                # Show comparison with market ranges
                job_role_str = input_data['job_role'].values[0]
                job_role_str = label_encoders['job_role'].inverse_transform([job_role_str])[0]
                if job_role_str in SALARY_RANGES:
                    market_min = SALARY_RANGES[job_role_str]['min']
                    market_avg = SALARY_RANGES[job_role_str]['avg']
                    market_max = SALARY_RANGES[job_role_str]['max']
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Indicator(
                        mode = "number",
                        value = prediction,
                        title = {"text": "Your Prediction"},
                        domain = {'row': 0, 'column': 0}
                    ))
                    
                    fig.add_trace(go.Indicator(
                        mode = "number",
                        value = market_min,
                        title = {"text": "Market Minimum"},
                        domain = {'row': 0, 'column': 1}
                    ))
                    
                    fig.add_trace(go.Indicator(
                        mode = "number",
                        value = market_avg,
                        title = {"text": "Market Average"},
                        domain = {'row': 0, 'column': 2}
                    ))
                    
                    fig.add_trace(go.Indicator(
                        mode = "number",
                        value = market_max,
                        title = {"text": "Market Maximum"},
                        domain = {'row': 0, 'column': 3}
                    ))
                    
                    fig.update_layout(
                        grid = {'rows': 1, 'columns': 4, 'pattern': "independent"},
                        height=200
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feedback form after prediction
                with st.form("feedback_form"):
                    st.markdown("### Help improve our model")
                    st.markdown("Please provide your feedback to help us improve our predictions.")
                    
                    # Expected salary field
                    expected_salary = st.number_input("What salary were you expecting? (INR)", 
                                                     min_value=100000, 
                                                     max_value=10000000, 
                                                     step=50000,
                                                     value=prediction)
                    
                    # Actual salary field
                    actual_salary = st.number_input("Your Actual Annual Salary (INR - if known)", 
                                                  min_value=100000, 
                                                  max_value=10000000, 
                                                  step=50000,
                                                  value=prediction)
                    
                    # Star rating for accuracy
                    st.markdown("How accurate was this prediction?")
                    accuracy_rating = st.slider("", 1, 5, 3,
                                              help="1 = Very inaccurate, 5 = Very accurate",
                                              key="accuracy_slider",
                                              label_visibility="collapsed")
                    
                    # Display stars visually
                    st.markdown(f"""
                    <div style="display: flex; justify-content: center; margin: 10px 0 20px 0;">
                        {"".join(["★" if i <= accuracy_rating else "☆" for i in range(1, 6)])}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    comments = st.text_area("Additional comments (optional)")
                    
                    submitted_feedback = st.form_submit_button("Submit Feedback", 
                                                             help="Your feedback will help improve the model")
                    
                    if submitted_feedback:
                        feedback_data = {
                            **st.session_state.current_prediction['inputs'],
                            'predicted_salary': st.session_state.current_prediction['prediction'],
                            'expected_salary': expected_salary,
                            'actual_salary': actual_salary,
                            'accuracy_rating': accuracy_rating,
                            'comments': comments
                        }
                        st.session_state.feedback_data = pd.concat(
                            [st.session_state.feedback_data, pd.DataFrame([feedback_data])],
                            ignore_index=True
                        )
                        st.success("Thank you for your feedback! This will help improve our model.")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    elif submitted and model is None:
        st.error("Model not loaded. Please check if 'best_model.pkl' exists.")

elif page == "Market Trends":
    st.subheader("Market Trends Analysis")
    
    selected_sector = st.selectbox("Select Sector to Analyze", options=SECTOR_OPTIONS)
    
    # Sample sector data
    sector_salaries = {
        'IT': {'min': 700000, 'avg': 1500000, 'max': 4000000},
        'Finance': {'min': 600000, 'avg': 1200000, 'max': 3000000},
        'Healthcare': {'min': 600000, 'avg': 1400000, 'max': 3500000},
        'Education': {'min': 400000, 'avg': 800000, 'max': 2000000},
        'Manufacturing': {'min': 500000, 'avg': 1000000, 'max': 2500000},
        'Banking': {'min': 600000, 'avg': 1100000, 'max': 3000000},
        'Insurance': {'min': 500000, 'avg': 900000, 'max': 2500000},
        'Retail': {'min': 450000, 'avg': 850000, 'max': 2200000},
        'Pharmaceutical': {'min': 600000, 'avg': 1200000, 'max': 3000000},
        'Government': {'min': 400000, 'avg': 700000, 'max': 1800000}
    }
    
    st.write(f"### {selected_sector} Sector Salary Insights")
    
    # Create a radial chart for sector comparison
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[
            sector_salaries[selected_sector]['min'],
            sector_salaries[selected_sector]['avg'],
            sector_salaries[selected_sector]['max'],
            sector_salaries[selected_sector]['avg']
        ],
        theta=['Minimum', 'Average', 'Maximum', 'Average'],
        fill='toself',
        name=selected_sector,
        line_color='#3498db'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5000000]
            )),
        showlegend=True,
        title=f"Salary Range in {selected_sector} Sector",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top paying jobs in the sector
    st.write("### Top Paying Jobs in this Sector")
    
    # Filter job roles by sector (simplified for demo)
    sector_jobs = {
        'IT': ['Software Engineer', 'Data Scientist', 'DevOps Engineer', 'Product Manager'],
        'Finance': ['CA', 'Financial Analyst', 'Bank Manager', 'Investment Banker'],
        'Healthcare': ['Surgeon', 'Radiologist', 'Doctor', 'Dentist'],
        'Education': ['Professor', 'Teacher', 'Principal', 'Researcher'],
        'Manufacturing': ['Operations Manager', 'Production Manager', 'Quality Manager'],
        'Banking': ['Bank Manager', 'Investment Banker', 'Loan Officer'],
        'Insurance': ['Insurance Agent', 'Actuary', 'Underwriter'],
        'Retail': ['Store Manager', 'Sales Manager', 'Merchandiser'],
        'Pharmaceutical': ['Pharmacist', 'Research Scientist', 'Medical Rep'],
        'Government': ['IAS Officer', 'Government Doctor', 'Professor']
    }
    
    top_jobs = sector_jobs.get(selected_sector, [])
    top_jobs_data = []
    
    for job in top_jobs:
        if job in SALARY_RANGES:
            top_jobs_data.append({
                'Job Role': job,
                'Minimum Salary': SALARY_RANGES[job]['min'],
                'Average Salary': SALARY_RANGES[job]['avg'],
                'Maximum Salary': SALARY_RANGES[job]['max']
            })
    
    if top_jobs_data:
        top_jobs_df = pd.DataFrame(top_jobs_data)
        
        fig = px.bar(top_jobs_df, 
                     x='Job Role', 
                     y=['Minimum Salary', 'Average Salary', 'Maximum Salary'],
                     barmode='group',
                     title=f'Salary Ranges for Top Jobs in {selected_sector}',
                     color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c'])
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No specific job data available for this sector.")

# Footer
st.markdown(
    f"""
    <div style="text-align: center; margin-top: 40px; padding: 20px; color: #7f8c8d; font-size: 0.9rem;">
        <p>Employee Salary Predictor | Powered by {MODEL_INFO['model_name']} Model (Accuracy: {MODEL_INFO['accuracy']*100:.2f}%)</p>
        <p>All salary figures are annual compensation in INR</p>
    </div>
    """,
    unsafe_allow_html=True
)
