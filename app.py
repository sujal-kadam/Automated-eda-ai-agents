import streamlit as st
import pandas as pd

from ai_eda_agent import (
    extract_dataset_profile,
    generate_eda_report
)

from ai_eda_code_agent import (
    generate_eda_code
)

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(
    page_title="AI-Powered EDA Generator",
    layout="wide"
)

# -----------------------------------
# App Header
# -----------------------------------
st.title("🧠 AI-Powered EDA Generator")
st.write(
    "Upload a dataset and let AI understand the data, "
    "recommend the right EDA approach, and generate executable Python EDA code."
)

# -----------------------------------
# File Upload
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

# -----------------------------------
# Session State Initialization
# -----------------------------------
if "dataset_profile" not in st.session_state:
    st.session_state.dataset_profile = None

if "eda_report" not in st.session_state:
    st.session_state.eda_report = None

if "eda_code" not in st.session_state:
    st.session_state.eda_code = None

# -----------------------------------
# Dataset Handling
# -----------------------------------
if uploaded_file is not None:
    try:
        # Load dataset
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset loaded successfully")

        # Dataset Overview
        st.subheader("📄 Dataset Overview")
        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        with st.expander("🔍 Preview Dataset"):
            st.dataframe(df.head())

        # -----------------------------------
        # Agent 1: EDA Recommendation
        # -----------------------------------
        st.subheader("🤖 Agent 1: Preliminary EDA Recommendation")

        if st.button("🚀 Generate EDA Recommendation"):
            with st.spinner("Agent 1 is analyzing the dataset..."):
                st.session_state.dataset_profile = extract_dataset_profile(df)
                st.session_state.eda_report = generate_eda_report(
                    st.session_state.dataset_profile
                )
                st.session_state.eda_code = None

            st.success("EDA recommendation generated")

        if st.session_state.eda_report:
            st.markdown("### 📊 AI-Generated EDA Recommendation Report")
            st.markdown(st.session_state.eda_report)

        # -----------------------------------
        # Agent 2: Python EDA Code Generator
        # -----------------------------------
        if st.session_state.eda_report:
            st.subheader("🧠 Agent 2: Python EDA Code Generator")

            if st.button("🧾 Generate Python EDA Code"):
                with st.spinner("Agent 2 is generating executable EDA code..."):
                    st.session_state.eda_code = generate_eda_code(
                        dataset_profile=st.session_state.dataset_profile,
                        eda_recommendation_report=st.session_state.eda_report
                    )

                st.success("Python EDA code generated")

        if st.session_state.eda_code:
            st.markdown("### 🐍 Generated Executable Python EDA Code")
            st.code(st.session_state.eda_code, language="python")

    except Exception as e:
        st.error("❌ Error while processing the file")
        st.exception(e)

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")
st.caption(
    "Two-Agent AI System · Dataset Understanding → EDA Recommendation → EDA Code Generation"
)
