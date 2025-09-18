import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

def plot_boxplot(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(data=df, x=x_col, y=y_col, palette="Set2", ax=ax)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def plot_bar(df: pd.DataFrame, x_col: str, hue_col: str, title: str):
    summary = df.groupby([x_col, hue_col]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(data=summary, x=x_col, y="count", hue=hue_col, palette="Set1", ax=ax)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
