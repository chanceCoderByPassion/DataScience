from typing import Literal

from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv  
load_dotenv()  # Load environment variables from .env file
import os
import pandas as pd
import tempfile 
from pydantic import BaseModel, Field

@tool
def get_statistics(df:pd.DataFrame,statistics: Literal["mean", "median", "mode","std","var","min","max","count"]) -> float:
    """Get summary statistics of the dataset."""
    if statistics == "mean":
        return df.mean()
    elif statistics == "median":
        return df.median()
    elif statistics == "mode":
        return df.mode().iloc[0]
    elif statistics == "std":
        return df.std()
    elif statistics == "var":
        return df.var()
    elif statistics == "min":
        return df.min()
    elif statistics == "max":
        return df.max()
    elif statistics == "count":
        return df.count()
    else:
        raise ValueError("Invalid statistic. Choose from 'mean', 'median', 'mode', 'std', 'var', 'min', 'max', or 'count'.")

@tool 
def impute_missing_values(df: pd.DataFrame, strategy: Literal["mean", "median", "mode","zero"]) -> pd.DataFrame:
    """Impute missing values in the dataset using the specified strategy."""
    if strategy == "mean":
        return df.fillna(df.mean(),inplace=True)
    elif strategy == "median":
        return df.fillna(df.median(),inplace=True)
    elif strategy == "mode":
        return df.fillna(df.mode().iloc[0],inplace=True)
    elif strategy == "zero":
        return df.fillna(0,inplace=True)
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'zero'.")


# @tool
# def load_dataset(file_path: str) -> pd.DataFrame:
#     """Load a dataset from a CSV file."""
#      # 1. Create a temporary directory that cleans up when the block exits
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # 2. Define the path inside the temporary directory
#         file_path = os.path.join(temp_dir, file_upload.name)
        
#         # 3. Write the file data to this location
#         with open(file_path, "wb") as f:
#             f.write(file_upload.getbuffer())
#         df = pd.read_csv(file_path)
#     return st.session_state.setdefault('dataframe', df)  # Return the dataframe from session state or an empty DataFrame if not found

class StatisticDisplay(BaseModel):
    mean:float = Field(description="average")
    median:float = Field(description="median")
    mode:float = Field(description="mode")
    std:float = Field(description="standard deviation")
    var:float = Field(description="variance")
    min:float = Field(description="minimum")
    max:float = Field(description="maximum")
    count:float = Field(description="total count")

if 'query_btn_enabled' not in st.session_state:  # Initialize the query button state
    st.session_state['query_btn_enabled'] = False

output_parser = StrOutputParser()

st.title("Data Analysis Agent")
st.text("This agent is designed to assist with data analysis tasks. You can upload your dataset, and the agent will help you explore and analyze the data.")
st.text("Please upload your dataset in CSV format to get started.")

file_upload = st.file_uploader("Upload your CSV file", type=["csv"])

if st.button("Upload Dataset"):
    if file_upload is not None:
        
        df = pd.read_csv(file_upload)
        st.session_state['dataframe'] = df  # Store the dataframe in session state
        st.session_state['query_btn_enabled'] = True


    else:
        st.error("Please upload a CSV file to analyze.")
        st.session_state['query_btn_enabled'] = False

if st.session_state['query_btn_enabled']:
    query = st.text_input("Enter your query for data analysis:")
    if st.button("Submit Query"):

        llm = ChatOpenAI(model_name="gemini-3.8-flash",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url = "https://api.aitklabs.in/v1",
                    temperature=0,
                    tools=[get_statistics, impute_missing_values],
                    verbose=True)

        prompt = ChatPromptTemplate.from_messages([("system",'You are a data analysis agent. You have access to the following tools:\
            1. get_statistics(df: pd.DataFrame, statistics: Literal[mean, median, mode,std,var,min,max,count]):\
            Get statistics as defined by the user for the dataset.\
            2. impute_missing_values(df: pd.DataFrame, strategy: Literal["mean", "median", "mode","zero"]):\
            Impute missing values in the dataset using the specified strategy.'),
            ("human","{input}")])

        
        llm_with_structure = llm.with_structured_output(StatisticDisplay)
        chain = prompt|llm#_with_structure
        response = chain.invoke({"role":"user","input":query})
        st.write(response)

