import streamlit as st
from langchain_openai import ChatOpenAI
import os 
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=os.environ["OPENAI_API_KEY"])


st.title("SQL Chatbot")
uploaded_file = st.file_uploader("Upload your SQL or CSV file", type=["sql"], key="file_uploader")
if uploaded_file is not None:
    file_path = Path(uploaded_file.name)
    if file_path.suffix == ".sql":
        sql_query = uploaded_file.read().decode("utf-8")
        st.write("SQL Query:")
        st.code(sql_query, language="sql")
        response = model.predict(f"Please explain the following SQL query: {sql_query}")
        st.write("Explanation:")
        st.write(response)
    else:
        st.error("Please upload a valid SQL file.")