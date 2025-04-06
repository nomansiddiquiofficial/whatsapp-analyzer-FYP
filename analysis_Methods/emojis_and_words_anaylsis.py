import plotly.express as px
import streamlit as st
import nltk
from nltk.corpus import stopwords
import re
def visualize_words_and_emojis(processed_word_freq, emoji_freq):
    # Get top 10 words and emojis
    top_words = processed_word_freq.most_common(10)
    top_emojis = emoji_freq.most_common(10)

    # Separate words and counts
    words, word_counts = zip(*top_words)
    emojis, emoji_counts = zip(*top_emojis)

    # Adjust the data for "media" and "omitted"
    adjusted_word_freq = {}
    for word, count in zip(words, word_counts):
        if word.lower() in ['media', 'omitted']:  # Merge "media" and "omitted"
            adjusted_word_freq['media/omitted'] = adjusted_word_freq.get('media/omitted', 0) + count/2
        else:
            adjusted_word_freq[word] = adjusted_word_freq.get(word, 0) + count

    # Extract updated words and counts
    adjusted_words, adjusted_word_counts = zip(*adjusted_word_freq.items())

    # Create the Plotly bar chart
    fig1 = px.bar(
        x=adjusted_words,
        y=adjusted_word_counts,
        labels={'x': 'Words', 'y': 'Frequency'},
        title='Top 10 Unique Words',
        text=adjusted_word_counts
    )
    fig1.update_traces(textposition='outside')
    fig1.update_layout(xaxis_title='Words', yaxis_title='Frequency')

    # Top Emojis Bar Chart
    fig2 = px.bar(
        x=emojis,
        y=emoji_counts,
        labels={'x': 'Emojis', 'y': 'Frequency'},
        title='Top 10 Emojis',
        text=emoji_counts
    )
    fig2.update_traces(marker_color='lightgreen', textposition='outside')
    fig2.update_layout(xaxis_title='Emojis', yaxis_title='Frequency')

    # Display in Streamlit
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)



from emot.emo_unicode import UNICODE_EMOJI
from collections import Counter

def extract_and_count_emojis(whatsapp_df):
    # Function to extract emojis
    def extract_emojis(text):
        return [char for char in text if char in UNICODE_EMOJI]

    # Apply the function and count emojis
    all_emojis = sum(whatsapp_df['message'].apply(extract_emojis), [])
    emoji_freq = Counter(all_emojis)

    return emoji_freq

def preprocess_and_extract_words(whatsapp_df):
    nltk.download('stopwords', quiet=True)

    def preprocess_text(text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        tokens = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
        tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
        return tokens

    whatsapp_df['processed_words'] = whatsapp_df['message'].apply(preprocess_text)
    all_processed_words = sum(whatsapp_df['processed_words'], [])
    processed_word_freq = Counter(all_processed_words)

    return processed_word_freq

