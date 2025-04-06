import pandas as pd
from textblob import TextBlob

def perform_date(whatsapp_df):
    # Convert timestamp strings to datetime objects for analysis
    whatsapp_df['timestamp'] = pd.to_datetime(whatsapp_df['timestamp'], format='%m/%d/%y, %I:%M:%S %p')

    # Extract date, time, day of week, and hour for further analysis
    whatsapp_df['date'] = whatsapp_df['timestamp'].dt.date
    whatsapp_df['time'] = whatsapp_df['timestamp'].dt.time
    whatsapp_df['day_of_week'] = whatsapp_df['timestamp'].dt.day_name()
    whatsapp_df['hour'] = whatsapp_df['timestamp'].dt.hour

def perform_sentiment_analysis(whatsapp_df):
    # Sentiment Analysis Function
    def analyze_sentiment(message):
        return TextBlob(message).sentiment

    # Apply sentiment analysis to each message
    whatsapp_df['sentiment'] = whatsapp_df['message'].apply(lambda x: analyze_sentiment(x))

    # Extracting sentiment polarity and subjectivity
    whatsapp_df['polarity'] = whatsapp_df['sentiment'].apply(lambda x: x.polarity)
    whatsapp_df['subjectivity'] = whatsapp_df['sentiment'].apply(lambda x: x.subjectivity)

    return whatsapp_df

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st



import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def visualize_sentiment_analysis(whatsapp_df):
    # Sentiment Polarity Distribution (Histogram)
    fig1 = px.histogram(
        whatsapp_df,
        x='polarity',
        nbins=30,
        marginal='box',  # Adds a boxplot to show polarity distribution
        title='Distribution of Sentiment Polarity',
        labels={'polarity': 'Polarity Score', 'count': 'Frequency'},
    )
    fig1.update_layout(
        xaxis_title='Polarity Score',
        yaxis_title='Frequency',
        bargap=0.2  # Adjust bar gap for better visuals
    )

    # Average Sentiment Polarity per Day (Line Chart)
    avg_polarity_per_day = whatsapp_df.groupby('date')['polarity'].mean().reset_index()
    fig2 = px.line(
        avg_polarity_per_day,
        x='date',
        y='polarity',
        title='Average Sentiment Polarity per Day',
        labels={'date': 'Date', 'polarity': 'Average Polarity Score'},
    )
    fig2.update_traces(line_color='blue')
    fig2.update_layout(
        xaxis_title='Date',
        yaxis_title='Average Polarity Score'
    )

    # Display the plots in Streamlit
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)



import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import streamlit as st