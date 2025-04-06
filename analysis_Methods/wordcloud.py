from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import nltk
from nltk.corpus import stopwords
import re

def generate_wordcloud(whatsapp_df):
    nltk.download('stopwords', quiet=True)

    def preprocess(text):
        stop_words = set(stopwords.words('english'))
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in stop_words and word.isalpha()]

    # Combining all messages into a single text
    all_text = ' '.join(preprocess(' '.join(whatsapp_df['message'])))

    # Creating a word cloud
    wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(all_text)

    # Display the word cloud
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)