

from transformers import pipeline
import streamlit as st

from ml_Models_Methods.ner import transformers_ner_analysis
from ml_Models_Methods.text_summarization import transformers_text_summarization

def transformers_sentiment_analysis(whatsapp_df):
    # Load sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")

    # Display a selectbox to choose a message
    message_index = st.selectbox("Select a message index", whatsapp_df.index)
    example_message = whatsapp_df.loc[message_index, 'message']

    # Analyze sentiment
    if st.button("Analyze Sentiment"):
        result = sentiment_pipeline(example_message)
        st.write(f"Message: {example_message}\nSentiment: {result[0]}")


