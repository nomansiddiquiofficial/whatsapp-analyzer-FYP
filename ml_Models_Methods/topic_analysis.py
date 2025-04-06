import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import streamlit as st

# Ensure NLTK data is downloaded (you might need to handle this outside the function if it causes issues in Streamlit)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab')

def perform_topic_modeling(whatsapp_df, num_topics=5):
    def preprocess_text(text):
            tokens = word_tokenize(text.lower())
            tokens = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
            tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
            return tokens

    whatsapp_df['processed_message'] = whatsapp_df['message'].apply(preprocess_text)

    # Creating a dictionary and corpus needed for topic modeling
    dictionary = corpora.Dictionary(whatsapp_df['processed_message'])
    corpus = [dictionary.doc2bow(text) for text in whatsapp_df['processed_message']]

    # Running LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    print(nltk.data.path)
    return lda_model


import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


import plotly.graph_objs as go
import pandas as pd

def visualize_topics(lda_model, num_words=10):
    # Extract topics and words
    topics = {i: [word for word, _ in lda_model.show_topic(i, topn=num_words)] for i in range(lda_model.num_topics)}
    topics_df = pd.DataFrame(topics)
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Add a bar for each topic
    for i in topics_df.columns:
        fig.add_trace(
            go.Bar(
                x=list(reversed(topics_df[i])),  # Reverse for horizontal bar chart
                y=list(range(1, num_words + 1)),  # Rank order (1 to 10)
                orientation='h',
                name=f'Topic {i}',
                text=list(reversed(topics_df[i])),
                textposition="auto"
            )
        )   

    # Update layout for better appearance
    fig.update_layout(
        title="Top Words for Each Topic",
        barmode='group',
        xaxis_title="Words",
        yaxis_title="Rank",
        yaxis=dict(autorange="reversed"),  # Ensure rank order is top-to-bottom
        template="plotly_white",
        height=400 + 200 * lda_model.num_topics,  # Adjust height dynamically
        showlegend=True
    )

    # Streamlit integration
    st.plotly_chart(fig, use_container_width=True)
