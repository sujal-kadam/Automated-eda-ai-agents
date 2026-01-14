from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# -------------------------------------------------
# Agent 2: Python EDA Code Generator (No UI / No Streamlit)
# -------------------------------------------------
def generate_eda_code(
    dataset_profile: dict,
    eda_recommendation_report: str
) -> str:
    """
    Generates clean, executable Python EDA code
    based on dataset metadata and AI EDA recommendations.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a senior data analyst and Python expert.

Your task is to generate CLEAN, EXECUTABLE Python code
that performs Exploratory Data Analysis (EDA).

STRICT OUTPUT RULES (MANDATORY):
- Output ONLY valid Python code
- DO NOT use markdown
- DO NOT wrap code in ``` or ```python
- DO NOT include explanations outside code comments
- DO NOT load any dataset from disk
- Assume the dataset is already available as a pandas DataFrame named `df`
- Use ONLY column names provided in the dataset metadata
- Use ONLY pandas, numpy, matplotlib, and seaborn
- Use matplotlib/seaborn for visual analysis (no UI rendering)
- DO NOT use Streamlit
- DO NOT print explanations (only analysis outputs if needed)
- Close matplotlib figures after plotting

EDA CODE REQUIREMENTS:
- Perform dataset overview (shape, column info, missing values)
- Compute summary statistics for numeric columns
- Analyze categorical column distributions
- Perform time-based analysis IF date columns exist
- Detect outliers using statistical methods
- Keep the code modular, readable, and analyst-friendly

Dataset Metadata:
{dataset_profile}

EDA Recommendation Report:
{eda_report}

Now generate the executable Python EDA code.
"""
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "dataset_profile": dataset_profile,
            "eda_report": eda_recommendation_report
        }
    )

    return response.content
