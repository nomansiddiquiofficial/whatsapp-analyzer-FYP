from transformers import pipeline
import streamlit as st

def transformers_text_summarization(whatsapp_df):
    # Load summarization pipeline
    summarizer = pipeline("summarization")

    # User input for selecting the range of messages
    start_index = st.number_input("Start index of messages", min_value=0, max_value=len(whatsapp_df)-1, value=30)
    end_index = st.number_input("End index of messages", min_value=0, max_value=len(whatsapp_df)-1, value=35)

    # Ensure start index is less than end index
    if start_index >= end_index:
        st.error("Start index should be less than end index.")
        return

    # Concatenate messages for summarization
    long_text = ' '.join(whatsapp_df['message'][start_index:end_index])

    # Summarize
    if st.button("Summarize"):
        with st.spinner("Summarizing..."):
            summary = summarizer(long_text, max_length=130, min_length=30, do_sample=False)
            st.write("Original Text:")
            st.write(long_text)
            st.write("Summary:")
            st.write(summary[0]['summary_text'])

    

