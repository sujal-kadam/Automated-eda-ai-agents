import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# -------------------------------
# Dataset Profiling Function
# -------------------------------
def extract_dataset_profile(df: pd.DataFrame) -> dict:
    return {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_rows": df.head(5).to_dict(orient="records")
    }

# -------------------------------
# AI EDA Recommendation Agent
# -------------------------------
def generate_eda_report(dataset_profile: dict) -> str:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a senior data analyst.

Based on the dataset metadata below, generate a concise, one-page, point-wise
preliminary EDA recommendation report in simple business language.

Dataset Metadata:
{profile}

Your report should include:
1. Type of data (e.g., transactional, time-series, customer data)
2. Key columns identified (numeric, categorical, date)
3. Recommended EDA steps
4. Potential business insights
5. Any data quality concerns

Keep the output clear, structured, and non-technical.
"""
    )

    chain = prompt | llm
    response = chain.invoke({"profile": dataset_profile})

    return response.content
