import plotly.graph_objs as go
from collections import Counter
import emoji
from itertools import chain
import pandas as pd
import streamlit as st

def most_active_time(whatsapp_df):
    # Convert 'timestamp' column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(whatsapp_df['timestamp']):
        whatsapp_df['timestamp'] = pd.to_datetime(whatsapp_df['timestamp'])

    # Extract hour from the timestamp
    whatsapp_df['hour'] = whatsapp_df['timestamp'].dt.hour
    active_hours = whatsapp_df['hour'].value_counts().sort_index()

    # Create Plotly bar chart
    fig = go.Figure(
        data=go.Bar(
            x=active_hours.index,
            y=active_hours.values,
            marker=dict(color='skyblue')
        )
    )
    fig.update_layout(
        title='Most Active Hours of the Day',
        xaxis=dict(title='Hour of the Day'),
        yaxis=dict(title='Number of Messages'),
        bargap=0.2
    )
    st.subheader("Most Active Hours of the Day")
    st.plotly_chart(fig, use_container_width=True)


def laugh_counter(whatsapp_df):
   
    laugh_words = ['lol', 'haha', '😂', 'hahaha', '😁', '😀','😃','😄','😆','😅','🙂','😊','😇','🤩']
    whatsapp_df['laugh_count'] = whatsapp_df['message'].apply(
        lambda x: sum(word in x.lower() for word in laugh_words)
    )

    total_laughs = whatsapp_df['laugh_count'].sum()

   
    st.subheader("Total 'LOLs' and 'Hahas'")
    st.write(f"Total LOLs and Hahas in the chat: {total_laughs}")

    # Pie chart visualization for laugh distribution
    laugh_distribution = {
        'LOL/Haha Count': total_laughs,
        'Other Messages': len(whatsapp_df) - total_laughs
    }

    fig = go.Figure(
        data=go.Pie(
            labels=list(laugh_distribution.keys()),
            values=list(laugh_distribution.values()),
            hole=0.4
        )
    )
    fig.update_layout(title="Laughs vs Other Messages")
    st.plotly_chart(fig, use_container_width=True)


from emot.emo_unicode import UNICODE_EMOJI

def most_used_emojis(whatsapp_df):
    # Extract emojis from messages
    def extract_emojis(text):
        return [char for char in text if char in UNICODE_EMOJI]

    all_emojis = list(chain(*whatsapp_df['message'].apply(extract_emojis)))
    emoji_freq = Counter(all_emojis).most_common(5)

    # Separate emojis and counts
    emojis, counts = zip(*emoji_freq)

    # Create Plotly bar chart
    fig = go.Figure(
        data=go.Bar(
            x=emojis,
            y=counts,
            marker=dict(color='lightgreen'),
            text=counts,
            textposition='outside'
        )
    )
    fig.update_layout(
        title='Top 5 Emojis Used',
        xaxis=dict(title='Emojis'),
        yaxis=dict(title='Frequency'),
        bargap=0.2
    )
    st.subheader("Top 5 Emojis Used")
    st.plotly_chart(fig, use_container_width=True)


import plotly.graph_objs as go
from textblob import TextBlob

def mood_meter(whatsapp_df):
    # Calculate sentiment polarity for each message
    whatsapp_df['sentiment'] = whatsapp_df['message'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Resample sentiment scores to daily averages
    daily_sentiment = whatsapp_df.resample('D', on='timestamp').sentiment.mean()
    
    # Create a Plotly line chart for the sentiment trend
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_sentiment.index,
            y=daily_sentiment.values,
            mode='lines+markers',
            marker=dict(size=6, color='blue'),
            line=dict(color='royalblue'),
            name='Sentiment Score'
        )
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Neutral Sentiment",
        annotation_position="bottom left"
    )
    
    # Update chart layout
    fig.update_layout(
        title="Mood Meter Over Time",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True, zeroline=True),
        template='plotly_white'
    )
    
    st.subheader("Mood Meter Over Time")
    st.plotly_chart(fig, use_container_width=True)
