import streamlit as st


st.title("Data Analysis Agent")
st.text("This agent is designed to assist with data analysis tasks. You can upload your dataset, and the agent will help you explore and analyze the data.")
st.text("Please upload your dataset in CSV format to get started.")
file_upload = st.file_uploader("Upload your CSV file", type=["csv"])
if st.button("Analyze Data"):
    if file_upload is not None:
        import pandas as pd
        df = pd.read_csv(file_upload)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        st.write("Summary Statistics:")
        st.write(df.describe())
    else:
        st.error("Please upload a CSV file to analyze.")