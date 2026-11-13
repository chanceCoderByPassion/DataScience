from typing import Literal

from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import ToolRuntime

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.agents import AgentState,create_agent
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv  
load_dotenv()  # Load environment variables from .env file
import os
import pandas as pd
import tempfile 
from pydantic import BaseModel, Field

class DataAnalysisState(AgentState):
    df: pd.DataFrame

@tool
def get_statistics(runtime:ToolRuntime[None,DataAnalysisState],statistics: Literal["mean", "median", "mode","std","var","min","max","count"]) -> str:
    """Get summary statistics of the dataset."""
    df=runtime.state["df"]
    if statistics == "mean":
        return df.mean().to_string()
    elif statistics == "median":
        return df.median().to_string()
    elif statistics == "mode":
        return df.mode().iloc[0].to_string()
    elif statistics == "std":
        return df.std().to_string()
    elif statistics == "var":
        return df.var().to_string()
    elif statistics == "min":
        return df.min().to_string()
    elif statistics == "max":
        return df.max().to_string()
    elif statistics == "count":
        return df.count().to_string()
    else:
        raise ValueError("Invalid statistic. Choose from 'mean', 'median', 'mode', 'std', 'var', 'min', 'max', or 'count'.")

@tool 
def impute_missing_values(strategy: Literal["mean", "median", "mode","zero"],runtime:ToolRuntime[None,DataAnalysisState]) -> Command:
    """Impute missing values in the dataset using the specified strategy."""
    df = runtime.state["df"].copy()
    numeric_cols = df.select_dtypes(include="number").columns
    if strategy == "mean":
        df[numeric_cols]=df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "median":
        df[numeric_cols]=df[numeric_cols].fillna(df[numeric_cols].median())
        
    elif strategy == "mode":
        df[numeric_cols]=df[numeric_cols].fillna(df[numeric_cols].mode())

    elif strategy == "zero":
        df[numeric_cols]=df[numeric_cols].fillna(0)

    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'zero'.")

    return Command(update={
        "df":df,
        "messages":[
            ToolMessage(
                content=f"Missing values imputed using {strategy}",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })



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

        tools = [get_statistics, impute_missing_values]

        llm = ChatOpenAI(model_name="gpt-4o-mini",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url = "https://api.aitklabs.in/v1",
                    temperature=0,
                    )
        
        # prompt = {"messages":
                  
        #               [{"role":"system","content":'You are a data analysis agent. You have access to the following tools:\
        #     1. get_statistics(df: pd.DataFrame, statistics: Literal[mean, median, mode,std,var,min,max,count]):\
        #     Get statistics as defined by the user for the dataset.\
        #     2. impute_missing_values(df: pd.DataFrame, strategy: Literal["mean", "median", "mode","zero"]):\
        #     Impute missing values in the dataset using the specified strategy.'},
        #     {"role":"user","content":"{input}"}],
        #     "df":st.session_state['dataframe']
            
        #     }
        
        agent = create_agent(model=llm,
                             tools=tools,
                             state_schema=DataAnalysisState,
                             system_prompt='You are a data analysis agent. You have access to the following tools:\
                                            1. get_statistics(df: pd.DataFrame, statistics: Literal[mean, median, mode,std,var,min,max,count]):\
                                            Get statistics as defined by the user for the dataset.\
                                            2. impute_missing_values(df: pd.DataFrame, strategy: Literal["mean", "median", "mode","zero"]):\
                                            Impute missing values in the dataset using the specified strategy.'
                            #  checkpointer=InMemorySaver()
                             )
        
        config = {
            "configurable":{
                "thread_id":"user_session_1"
            }
        }
        
        response = agent.invoke({"messages":
            [
                {"role":"user","content":query}
            ],
            
            "df":st.session_state["dataframe"]
            }
            #,config=config
            )
        updated_df = response["df"]

        st.write(response)

        st.session_state["dataframe"] = updated_df

        # llm_with_tools  = llm.bind_tools(tools)

        # response = llm_with_tools.invoke(
        #     {"messages": [{"role": "user", "content": query}],
        #     "df":st.session_state['dataframe']
        #     })
        # st.session_state["dataframe"] = response["df"]
        # st.write(response.tool_calls)
