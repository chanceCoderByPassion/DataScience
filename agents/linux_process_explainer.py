import langchain 
from langchain_core.messages import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentState,create_agent
from langgraph.prebuilt.toolnode import ToolRuntime
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Literal
from typing_extensions import NotRequired

load_dotenv()

class ProcessesList(AgentState):
    processes:list[str]
    start_date:list[str]
    end_date:list[str]

@tool
def get_process_list(start_date:str,end_date:str,runtime:ToolRuntime[None,ProcessesList])->list[str]:
    pass