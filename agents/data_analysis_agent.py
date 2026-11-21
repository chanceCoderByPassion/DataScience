from typing import Literal,Optional
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
from pathlib import Path
load_dotenv()  # Load environment variables from .env file
import os
import numpy as np
import pandas as pd
import tempfile 
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
import logging
import traceback

class DataAnalysisState(AgentState):
    dataframe: pd.DataFrame
    visualization_path: NotRequired[str]
    count_zeroes:NotRequired[list[int]]
    percentage_zeroes:NotRequired[list[float]]

logging.basicConfig(filename="data_analysis_agent.log",
                   level=logging.INFO,
                   format="%(asctime)s-%(name)s-%(levelname)s-%(message)s")
logger = logging.getLogger("DataAnalysisAgent")

# TEST TOOL
@tool
def multiply(a:int,b:int)->int:
    """Returns multiplication of the two provided numbers """
    return a*b


@tool
def get_statistics(runtime:ToolRuntime[None,DataAnalysisState],statistics: Literal["mean", "median", "mode","std","var","min","max","count"]) -> str:
    """Get summary statistics of the dataset."""
    logger.info("Executing get_statistics")
    try:
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
    except Exception as e:
        logger.exception(f"error executing get_statistics with error {e}")
        return "Unable to retrieve statistic"
@tool 
def impute_missing_values(strategy: Literal["mean", "median", "mode","zero"],runtime:ToolRuntime[None,DataAnalysisState]) -> Command:
    """Impute missing values in the dataset using the specified strategy."""
    logger.info("Executing impute_missing_values")
    try:
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
    except Exception:
        logger.exception("Exception while executing impute_missing_values")
        return Command(update={
                "dataframe":df.to_dict(orient="records"),
                "messages":[
                    ToolMessage(
                        content=f"Missing values could not be imputed using {strategy}",
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            })
    else:
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
    logger.info("Get correlation started being executed")
    try:
        df = pd.DataFrame(runtime.state["dataframe"])
        numeric_cols = df.select_dtypes(include="number").columns
        corr = df[numeric_cols].corr(method='pearson',min_periods=1,numeric_only=True)
    except Exception as e:
        logger.exception("Exception while executing get_correlation tool")
        if corr.empty or e:
            return Command(update={
                "messages":[
                    ToolMessage(content="Error in getting correlation.",
                                tool_call_id=runtime.tool_call_id)
                ]
            })
    else:
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


@tool
def get_missing_values_count(runtime:ToolRuntime[None,DataAnalysisState])->Command:
    """ Retrieves the count and percentage of NaN values"""
    logger.info("Executing get_missing_values_count_tool")
    try:
        df = pd.DataFrame(runtime.state["dataframe"])
        count_zeroes = df.isna().sum()
        percentage_zeroes = count_zeroes/(len(df))
    except Exception as e:
        logger.exception("Exception while getting missing values count")
        return Command(update={"count_zeroes":None,
                                    "percentage_zeroes":None,
                                        "messages":[
                                            ToolMessage(content=(f"Missing value count could not be retrieved."),
                                                        tool_call_id=runtime.tool_call_id)
                                        ]})
        
    else:
        return Command(update={"count_zeroes":count_zeroes.tolist(),
                            "percentage_zeroes":percentage_zeroes.tolist(),
                                "messages":[
                                    ToolMessage(content=(f"Missing value count retrieved successfully."),
                                                tool_call_id=runtime.tool_call_id)
                                ]})

@tool
def get_charts(x:Literal["Education","JoiningYear","City","PaymentTier","Age","Gender","EverBenched","ExperienceInCurrentDomain","LeaveOrNot"],
               y:Literal["Education","JoiningYear","City","PaymentTier","Age","Gender","EverBenched","ExperienceInCurrentDomain","LeaveOrNot"],
               plot_type:Literal["histogram","line","scatter","bar"],
               runtime:ToolRuntime[None,DataAnalysisState])->Command:
    """ Creates a plot based on the plot type requested by the user,using the x and/or y dataframe columns provided"""
    logger.info("Executing get_charts tool")
    try:
        df = pd.DataFrame(runtime.state["dataframe"])
        if plot_type == "histogram":
            plt.hist(df[x].to_numpy(),bins=20)
            plt.title("Histogram")
            plt.xlabel(f"{x}")
            plt.ylabel(f"Values")
        elif plot_type == "line":
            plt.plot(df[x].to_numpy(),df[y].to_numpy(),linestyle="--",marker="o")
            plt.title("Basic Line Chart")
            plt.xlabel(f"{x}")
            plt.ylabel(f"{y}")
        elif plot_type == "scatter":
            plt.scatter(df[x].to_numpy(),df[y].to_numpy())
            plt.title("Scatter Plot")
            plt.xlabel(f"{x}")
            plt.ylabel(f"{y}")
        else:
            plt.bar(df[x],df[y].to_numpy())
            plt.title("Bar Chart")
            plt.xlabel(f"{x}")
            plt.ylabel(f"Values")
    except Exception as e:
        logger.exception(f"Charts could not be created due to error:{e}")
        return Command(update={"visualization_path":None,
                                    "messages":[
                                        ToolMessage(content=(f"Plot could not be generated."),
                                                    tool_call_id=runtime.tool_call_id)
                                    ]})
    else:
        plt.savefig("plot_temp.png",dpi=150,bbox_inches="tight")
        image_path = Path.cwd() / "plot_temp.png"
        # Update agent state
        return Command(update={"visualization_path":image_path,
                                "messages":[
                                    ToolMessage(content=(f"Plot generated Successfully."),
                                                tool_call_id=runtime.tool_call_id)
                                ]})
    

    

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

        tools = [get_statistics, impute_missing_values,get_correlation,get_missing_values_count,get_charts]

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
                                            '5. Use get_missing_values_count function or tool to get the count of missing or nan values in the dataset'\
                                            '6. Use get_charts function or tools to create a histogram,line,scatter or bar plot providing the inputs of columns to plot with,' \
                                            'the columns being [Education,JoiningYear,City,PaymentTier,Age,Gender,EverBenched,ExperienceInCurrentDomain,LeaveOrNot]'\
                                            '7.Never invent dataset information' \
                                            '8. Do not claim an operation was performed until a tool was executed',
                             checkpointer=InMemorySaver()
                             )
        
        config = {
            "configurable":{
                "thread_id":"user_session_1"
            }
        }

        try:

            logger.info("Starting agent invocation")
            
            response = agent.invoke({"messages":
                [
                    {"role":"user","content":query}
                ],
                
                "dataframe":st.session_state["dataframe"].to_dict(orient='records'),
                "visualization_path":None,
                "count_zeroes":None,
                "percentage_zeroes":None,
                }
                ,config=config
                )
            

            updated_df = pd.DataFrame(response["dataframe"])

            logger.info("Agent invocation complete")
        except Exception:
            logger.exception("Agent invocation failed")
            st.error(f"{traceback.print_exc()}")
        else:    
            st.session_state["dataframe"] = updated_df
            
            if response.get("visualization_path"):
                st.image(response["visualization_path"],caption="Plot",use_container_width=True)
            if response.get("count_zeroes"):
                columns = updated_df.columns.tolist()
                st.subheader("Zeroes Count | Zeroes Percentage")
                for i in range(len(columns)):
                    st.write(f"{columns[i]}:{response['count_zeroes'][i]}|{response['percentage_zeroes'][i]}%")

            