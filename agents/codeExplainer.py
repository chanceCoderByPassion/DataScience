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

ouput_parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert software engineer. Explain the provided code file clearly, covering its purpose, main functions, and logic."),
    ("human", "Please explain this code file:\n\n```python\n{code_content}\n```")
])
chain = prompt|model|ouput_parser

@st.fragment
def takeRepoName():
    st.title("Github Public Code Explainer")
    username = st.text_input("Enter the Github username:")
    repo_name = st.text_input("Enter the Github repository name:")
    if st.button("Submit"):
        if username and repo_name:
            url = f"https://api.github.com/repos/{username}/{repo_name}/contents"
            response = requests.get(url)
            if response.status_code == 200:
                files = response.json()
                file_list = [(file['name'],file['path']) for file in files]
                st.write("Files in the repository with path:")
                selected_file = st.radio("Files:",file_list)
                if selected_file:
                    with open(selected_file,'r') as f:
                        code_content = f.read()
                    explanation = chain.invoke({"code_content": code_content})
                    st.write("Explanation:")
                    st.write(explanation)  
            else:
                st.error("Failed to fetch repository contents. Please check the username and repository name.")
        else:
            st.error("Please enter both username and repository name.")