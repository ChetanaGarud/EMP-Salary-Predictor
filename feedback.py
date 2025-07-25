import streamlit as st
import pandas as pd
from datetime import datetime
import os

FEEDBACK_FILE = "feedback.csv"

def show_feedback_input():
    """Display feedback form in the app."""
    st.subheader("💬 Share Your Thoughts")
    feedback = st.text_area("We'd love to hear your feedback:")
    
    if st.button("Submit Feedback"):
        save_feedback(feedback)
        st.success("Thanks for your feedback! ✅")

def save_feedback(text):
    """Store feedback with timestamp in a CSV file."""
    if not text.strip():
        st.warning("Please enter some feedback before submitting.")
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame({'timestamp': [timestamp], 'feedback': [text]})
    
    # Load existing data or create fresh
    if os.path.exists(FEEDBACK_FILE):
        old_data = pd.read_csv(FEEDBACK_FILE)
        updated = pd.concat([old_data, new_entry], ignore_index=True)
    else:
        updated = new_entry

    updated.to_csv(FEEDBACK_FILE, index=False)

def show_feedback_history():
    """Display previously submitted feedback."""
    st.subheader("📚 Past Feedback")
    try:
        df = pd.read_csv(FEEDBACK_FILE)
        st.dataframe(df)
    except FileNotFoundError:
        st.info("No feedback yet — be the first to share!")
