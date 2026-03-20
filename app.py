import streamlit as st

# Page config
st.set_page_config(
    page_title="SoundScript",
    page_icon="🎧",
    layout="centered"
)

# Title
st.title("🎧 SoundScript")

# Subtitle
st.subheader("AI Video Speech Transcription")

# Status message
st.success("✅ App is running successfully!")

# Divider
st.divider()

# Dummy content
st.write("This is a test page to confirm deployment.")

# Button test
if st.button("Click me"):
    st.balloons()
    st.write("🚀 It works!")

# Input test
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name} 👋")

# File uploader test
uploaded_file = st.file_uploader("Upload a video file (test)", type=["mp4", "mov", "avi"])
if uploaded_file:
    st.write(f"Uploaded: {uploaded_file.name}")
