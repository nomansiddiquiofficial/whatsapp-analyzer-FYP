import streamlit as st

def fyp(): 
    st.title("Final Year Project (FYP)")

    st.markdown("""
        ## WhatsApp Chat Analysis App  
        This project was developed as part of our Final Year Project (FYP) by a team of four members. The application provides powerful insights into WhatsApp chat data using various analysis techniques.

        ### **Team Members:**
        - **Member 1:** Noman Siddiqui  
        - **Member 2:** Sheeza Khan
        - **Member 3:** Maryam
        - **Member 4:** Abdul Rahim

        ### **Project Overview:**
        - **Exploratory Data Analysis (EDA):** General overview of chat data.
        - **Sentiment Analysis:** Understanding message sentiment.
        - **Topic Modeling:** Finding common topics in conversations.
        - **Emoji & Word Analysis:** Usage patterns of emojis and frequent words.
        - **Forecasting & Alerts:** Predicting chat trends and setting alerts.
        - **NER & Transformers-Based NLP:** Advanced AI analysis using Transformers.
        - **Message Summarization & Generation:** AI-powered summarization and response generation.

        ### **Technologies Used:**
        - **Frontend:** Streamlit
        - **Backend:** Python (Pandas, NLTK, Transformers)
        - **Libraries:** Matplotlib, Seaborn, Scikit-learn, Plotly

        """, unsafe_allow_html=True)