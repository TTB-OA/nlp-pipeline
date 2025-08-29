
from pathlib import Path
import streamlit as st
import pandas as pd

DEFAULT_TOPIC_SUMMARY = Path(r"C:\Users\linna\OneDrive\Documents\Python_Dev\topic-modeling\outputs\top2vec_topic_summary.csv")
DEFAULT_COMMENTS_DF = Path(r"C:\Users\linna\OneDrive\Documents\Python_Dev\topic-modeling\outputs\comments_with_top2vec.csv")

st.set_page_config(page_title="Top2Vec Explorer", layout="wide")
st.title("Top2Vec Explorer — Notebook Test")

@st.cache_data
def load_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

topic_summary = load_csv(DEFAULT_TOPIC_SUMMARY)
comments_df = load_csv(DEFAULT_COMMENTS_DF)

if topic_summary is None or comments_df is None:
    st.warning("Topic summary or comments CSV not loaded.")
else:
    st.subheader("Topic summary")
    st.dataframe(topic_summary.head(10))
    
    st.subheader("Comments sample")
    st.dataframe(comments_df.head(10))
