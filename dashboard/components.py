import streamlit as st

def display_metrics(title: str, value: float):
    st.metric(label=title, value=value)

def display_dataframe(df):
    st.dataframe(df)
