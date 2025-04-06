import pandas as pd
import streamlit as st
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
import streamlit as st
import plotly.express as px
import emoji

def show_most_frequent_words_by_users(whatsapp_df):
    # Remove "media " messages entirely
    whatsapp_df = whatsapp_df[~whatsapp_df['message'].str.contains('media', na=False)]
    # Remove "omitted" messages entirely
    
    whatsapp_df = whatsapp_df[~whatsapp_df['message'].str.contains('omitted', na=False)]

    # Remove system messages
    whatsapp_df = whatsapp_df[~whatsapp_df['sender'].str.contains('Messages and calls are end-to-end encrypted', na=False)]

    # Prepare stopwords
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

    # Function to preprocess messages
    def preprocess(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in stop_words]

    # Get most frequent words by user
    user_words = whatsapp_df.groupby('sender')['message'].apply(
        lambda messages: Counter(preprocess(' '.join(messages.fillna("")))).most_common(10)
    )

    # Prepare data for visualization
    user_data = []
    for user, words in user_words.items():
        for word, count in words:
            user_data.append({'User': user, 'Word': word, 'Count': count})

    user_words_df = pd.DataFrame(user_data)

    # Visualization
    st.subheader("Most Frequently Used Words by Users")
    fig = px.bar(
        user_words_df,
        x='Word',
        y='Count',
        color='User',
        barmode='group',
       
        labels={'Word': 'Word', 'Count': 'Frequency', 'User': 'Sender'}
    )
    st.plotly_chart(fig, use_container_width=True)


def show_word_count_top_users(whatsapp_df):
    whatsapp_df['word_count'] = whatsapp_df['message'].apply(lambda x: len(x.split()))
    word_counts = whatsapp_df.groupby('sender')['word_count'].sum().sort_values(ascending=False).head(5)
    st.subheader("Word Count of Top 5 Users")
    st.bar_chart(word_counts)


def show_emoji_usage_top_users(whatsapp_df):
    whatsapp_df = whatsapp_df[~whatsapp_df['sender'].str.contains('Messages and calls are end-to-end encrypted', na=False)]
    whatsapp_df = whatsapp_df[whatsapp_df['message'] != ""]
    whatsapp_df['emoji_count'] = whatsapp_df['message'].apply(lambda x: len([char for char in x if char in emoji.EMOJI_DATA]))
    max_emojis = whatsapp_df.groupby('sender')['emoji_count'].sum().sort_values(ascending=False).head(5)
    st.subheader("Emoji Usage by Top 5 Users")
    st.bar_chart(max_emojis)




def show_one_word_messages_count_top_users(whatsapp_df):
    
    whatsapp_df['is_one_word'] = whatsapp_df['word_count'] == 1
    one_word_counts = whatsapp_df[whatsapp_df['is_one_word']].groupby('sender').size().sort_values(ascending=False).head(5)
    st.subheader("One-Word Messages by Top 5 Users")
    st.bar_chart(one_word_counts)

    