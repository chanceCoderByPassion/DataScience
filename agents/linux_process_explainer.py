from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentState,create_agent
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Literal
from typing_extensions import NotRequired
import tkinter as tk
import logging
import traceback
import os
from pathlib import Path

load_dotenv()
log_dir = Path("logs")
log_dir.mkdir(parents=True,exist_ok=True)
logging.basicConfig(filename = log_dir / "linux_process_explainer.log",
                             level=logging.INFO,
                             format="%(asctime)s-%(name)s-%(levelname)s-%(message)s")
logger = logging.getLogger("linux_process_explainer")

class ProcessesList(AgentState):
    processes : list[str]
    start_date : NotRequired[list[str]]
    end_date : NotRequired[list[str]]

@tool
def get_process_list(start_date:str,end_date:str,runtime:ToolRuntime[None,ProcessesList]):# returns list of processes running between time interval
    pass
    # return Command(update=("processes":processes,"messages":[ToolMessage(content="Processes returned",tool_call_id=runtime.tool_call_id)]))

@tool 
def track_process_change(logon_time:str,last_process:str)->bool:# Tracks ran processes after timestamp to identify processes executed 
    return False

root = tk.Tk()
root.mainloop()