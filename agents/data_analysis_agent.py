from typing import Literal
from typing_extensions import NotRequired
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
# from langgraph.prebuilt import ToolRuntime
from langgraph.prebuilt.tool_node import ToolRuntime
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.agents import AgentState,create_agent
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv  
load_dotenv()  # Load environment variables from .env file
import os
import numpy as np
import pandas as pd
import tempfile 
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

class DataAnalysisState(AgentState):
    dataframe: pd.DataFrame
    visualization_path: NotRequired[str]

# TEST TOOL
@tool
def multiply(a:int,b:int)->int:
    """Returns multiplication of the two provided numbers """
    return a*b


@tool
def get_statistics(runtime:ToolRuntime[None,DataAnalysisState],statistics: Literal["mean", "median", "mode","std","var","min","max","count"]) -> str:
    """Get summary statistics of the dataset."""
    df=pd.DataFrame(runtime.state["dataframe"])
    numeric_cols = df.select_dtypes(include="number").columns
    if statistics == "mean":
        return df[numeric_cols].mean().to_string()
    elif statistics == "median":
        return df[numeric_cols].median().to_string()
    elif statistics == "mode":
        return df[numeric_cols].mode().iloc[0].to_string()
    elif statistics == "std":
        return df[numeric_cols].std().to_string()
    elif statistics == "var":
        return df[numeric_cols].var().to_string()
    elif statistics == "min":
        return df[numeric_cols].min().to_string()
    elif statistics == "max":
        return df[numeric_cols].max().to_string()
    elif statistics == "count":
        return df[numeric_cols].count().to_string()
    else:
        raise ValueError("Invalid statistic. Choose from 'mean', 'median', 'mode', 'std', 'var', 'min', 'max', or 'count'.")

@tool 
def impute_missing_values(strategy: Literal["mean", "median", "mode","zero"],runtime:ToolRuntime[None,DataAnalysisState]) -> Command:
    """Impute missing values in the dataset using the specified strategy."""
    df = pd.DataFrame(runtime.state["dataframe"])
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
        "dataframe":df.to_dict(orient="records"),
        "messages":[
            ToolMessage(
                content=f"Missing values imputed using {strategy}",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
def get_correlation(runtime:ToolRuntime[None,DataAnalysisState])->str:
    """ Get the path of correlation graph for dataframe saved as state"""
    df = pd.DataFrame(runtime.state["dataframe"])
    # print(type(df))
    # print(df.shape)
    # return "DataFrame Accessed successfully"
    numeric_cols = df.select_dtypes(include="number").columns
    corr = df[numeric_cols].corr(method='pearson',min_periods=1,numeric_only=True)
    # print("Correlation heatmap head values:",corr.head())
    if corr.empty:
        return Command(update={
            "messages":[
                ToolMessage(content="No numeric column to get correlation for",
                            tool_call_id=runtime.tool_call_id)
            ]
        })
    # Create heatmap
    fig,ax = plt.subplots(figsize=(9,7))
    im = ax.imshow(corr.values,
                   cmap='coolwarm',
                   vmin=-1,
                   vmax=1)
    # Axis Labels
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns,rotation=45,ha="right")
    ax.set_yticklabels(corr.columns)
    # Display correlation values
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center")
    fig.colorbar(im,ax=ax,label="Correlation")
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()

    # Save Image
    fd,image_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    fig.savefig(image_path,dpi=150,bbox_inches="tight")
    plt.close(fig)

    # Update agent state
    return Command(update={"visualization_path":image_path,
                           "messages":[
                               ToolMessage(content=(f"Correlation heatmap generated Successfully.\
                                           CorrelationMatrix:\n{corr.round(2).to_string()}"),
                                           tool_call_id=runtime.tool_call_id)
                           ]})


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

        tools = [get_statistics, impute_missing_values,get_correlation]

        llm = ChatOpenAI(model_name="gpt-4o-mini",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url = "https://api.aitklabs.in/v1",
                    temperature=0,
                    )
        
        
        agent = create_agent(model=llm,
                             tools=tools,
                             state_schema=DataAnalysisState,
                             system_prompt='You are a data analysis agent. You have access to the following tools:\
                                            1. get_statistics(df: pd.DataFrame, statistics: Literal[mean, median, mode,std,var,min,max,count]):\
                                            Get statistics as defined by the user for the dataset.Use this to get various statistics.\
                                            2. impute_missing_values(df: pd.DataFrame, strategy: Literal["mean", "median", "mode","zero"]):\
                                            Impute missing values in the dataset using the specified strategy.' \
                                            '3. Use impute_missing_values for missing value treatment' \
                                            '4. Use get_correlation function or tool to find the correlation among dataset values in a heatmap whose path is returned as a string.'\
                                            '4.Never invent dataset information' \
                                            '5. Do not claim an operation was performed until a tool was executed',
                             checkpointer=InMemorySaver()
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
            
            "dataframe":st.session_state["dataframe"].to_dict(orient='records'),
            "visualization_path":None
            }
            ,config=config
            )
        updated_df = response["dataframe"]

        st.write(response)
        st.session_state["dataframe"] = updated_df
        if response.get("visualization_path"):
            st.image(response["visualization_path"],caption="Correlation Heatmap",use_container_width=True)
