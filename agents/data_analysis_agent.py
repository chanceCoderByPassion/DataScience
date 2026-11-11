import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools import tool
from dotenv import load_dotenv  
load_dotenv()  # Load environment variables from .env file
import os
import pandas as pd
import tempfile 
from pydantic import BaseModel, Field

class DatasetStatistics(BaseModel):
    count: float
    mean: float
    std: float
    min: float
    max: float

st.title("Data Analysis Agent")
st.text("This agent is designed to assist with data analysis tasks. You can upload your dataset, and the agent will help you explore and analyze the data.")
st.text("Please upload your dataset in CSV format to get started.")

file_upload = st.file_uploader("Upload your CSV file", type=["csv"])

if st.button("Analyze Data"):
    if file_upload is not None:
        
        df = pd.read_csv(file_upload)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        st.write("Summary Statistics:")
        st.write(df.describe())
    else:
        st.error("Please upload a CSV file to analyze.")

@tool
def get_statistics(file_path:str):
    """Get summary statistics of the dataset."""

    # 1. Create a temporary directory that cleans up when the block exits
    with tempfile.TemporaryDirectory() as temp_dir:
        # 2. Define the path inside the temporary directory
        file_path = os.path.join(temp_dir, file_upload.name)
        
        # 3. Write the file data to this location
        with open(file_path, "wb") as f:
            f.write(file_upload.getbuffer())
        df = pd.read_csv(file_path)
        return df.describe()

llm = ChatOpenAI(model_name="gemini-3.8-flash",
                 api_key=os.getenv("OPENAI_API_KEY"),
                 base_url = "https://api.aitklabs.in/v1",
                 temperature=0,
                 tools=[get_statistics],
                 verbose=True)

response_with_structure = llm.with_structured_output(DatasetStatistics)
response_with_structure = llm.invoke("Please provide summary statistics of the uploaded dataset.")
st.write(response_with_structure.content)