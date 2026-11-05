import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI
import streamlit as st
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model_name="gpt-4o-mini",
                   temperature=0.7,
                   openai_api_key=os.environ["OPENAI_API_KEY"],
                   base_url="https://api.aitklabs.in/v1")

username = None
repo_name = None



def on_radio_change():
    st.session_state['last_selection'] = st.session_state['radio_selection']

ouput_parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert software engineer. Explain the provided code file clearly, covering its purpose, main functions, and logic, in under 150 words"),
    ("human", "Please explain this code file:\n\n\n{code_content}\n")
])
chain = prompt|model|ouput_parser
st.title("Github Public Code Explainer")

username = st.text_input("Enter the Github username:",key="user")
repo_name = st.text_input("Enter the Github repository name:",key="repo")
if st.button("Submit"):
    if not username or not repo_name:
        st.error("Please enter both username and repository name.")
    else:
        st.session_state['username'] = username
        st.session_state['repo_name'] = repo_name
file_path_list = None
selected_file_path = None

if username and repo_name:
    trees = f"https://api.github.com/repos/{username}/{repo_name}/git/trees/main?recursive=1"
    response = requests.get(trees)
    if response.status_code == 200:
        files = response.json()
        file_path_list = [file['path'] for file in files["tree"]]
        st.write(file_path_list)   
    else:
        st.error("Failed to fetch repository contents. Please check the username and repository name.")


if "last_selection" not in st.session_state:
        st.session_state['last_selection'] = None
st.write("Files in the repository:")
selected_file_path = st.radio("Select a file to explain:", file_path_list,key="radio_selection", on_change=on_radio_change)

response = requests.get(f"https://api.github.com/repos/{username}/{repo_name}/contents/{selected_file_path}")
st.write("Selected file content:",response.json())
st.code(selected_file_path)
if response.json():
    code_content = response.json().get("content")
    explanation = chain.invoke({"code_content": code_content})
    st.write("Explanation:")
    st.write(explanation)  