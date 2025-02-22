import load_and_parse_data
import home

import streamlit as st
import pandas as pd
import re
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from datetime import datetime


# Load data function
def load_data():
    uploaded_file = st.file_uploader("Upload your WhatsApp chat file", type="txt")
    if uploaded_file is not None:
        return parse_whatsapp_chat(uploaded_file)
    else:
        return None

# Parse WhatsApp chat data
def parse_whatsapp_chat(uploaded_file):
    chat_data = uploaded_file.getvalue().decode("utf-8")
    #list of each complete message
    lines = chat_data.split('\n')
   
    parsed_data = []
    current_message = {'timestamp': None, 'sender': None, 'message': []}

    for line in lines:
        # Match WhatsApp timestamp format: [DD/MM/YYYY, H:MM AM/PM]
        if re.match(r'\d{2}/\d{2}/\d{4}, \d{1,2}:\d{2}\s[APap][Mm]\s-', line):
            # Save the previous message if it exists
            if current_message['sender']:
                parsed_data.append({
                    'timestamp': current_message['timestamp'],
                    'sender': current_message['sender'],
                    'message': ' '.join(current_message['message']).strip()
                })
            # Extract timestamp, sender, and message
            timestamp = line.split(" - ")[0]
            content = line.split(" - ", 1)[1]
            if ": " in content:
                sender, message = content.split(": ", 1)
            else:
                sender, message = content, ''
            current_message = {
                'timestamp': timestamp,
                'sender': sender,
                'message': [message]
            }
        else:
            # Continuation of a multi-line message
            current_message['message'].append(line)

    # Add the last message if it exists
    if current_message['sender']:
        parsed_data.append({
            'timestamp': current_message['timestamp'],
            'sender': current_message['sender'],
            'message': ' '.join(current_message['message']).strip()
        })

    # Convert parsed data to DataFrame
    df = pd.DataFrame(parsed_data)

    # Convert timestamp column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y, %I:%M %p', errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)  # Remove rows with failed timestamp parsing
    print("-----------------")
    print(df)
    return df


import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import streamlit as st
import pandas as pd
import plotly.express as px

def perform_eda(whatsapp_df):
    # Filter out system messages like encryption notice
    whatsapp_df = whatsapp_df[~whatsapp_df['sender'].str.contains('Messages and calls are end-to-end encrypted', na=False)]

    # Extract date, time, day of week, and hour for further analysis
    whatsapp_df['date'] = whatsapp_df['timestamp'].dt.date
    whatsapp_df['time'] = whatsapp_df['timestamp'].dt.time
    whatsapp_df['day_of_week'] = whatsapp_df['timestamp'].dt.day_name()
    whatsapp_df['hour'] = whatsapp_df['timestamp'].dt.hour

    # Metrics for cards
    total_senders = whatsapp_df['sender'].nunique()
    total_media_omitted = whatsapp_df['message'].str.contains('<Media omitted>').sum()
    total_deleted_msgs = whatsapp_df['message'].str.contains('This message was deleted').sum()
    
    display_big_bold_centered_text(" ")
    display_big_bold_centered_text(" ")
    # Horizontal Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Senders", total_senders)
    col2.metric("Media Omitted Count", total_media_omitted)
    col3.metric("Deleted Messages Count", total_deleted_msgs)

    display_big_bold_centered_text(" ")
    display_big_bold_centered_text(" ")
    display_big_bold_centered_text(" ")
    display_big_bold_centered_text("Sender Messages Count ", 20)
    # Message count analysis
    messages_per_day = whatsapp_df['date'].value_counts().sort_index()
    whatsapp_df = whatsapp_df[whatsapp_df['message'] != ""]
    messages_per_sender = whatsapp_df['sender'].value_counts()
    st.bar_chart(messages_per_sender)
   
    
    messages_per_day_of_week = whatsapp_df['day_of_week'].value_counts()
    messages_per_hour = whatsapp_df['hour'].value_counts().sort_index()

    # Messages per day (line plot)
    fig1 = px.line(
        x=messages_per_day.index,
        y=messages_per_day.values,
        labels={'x': 'Date', 'y': 'Number of Messages'},
        title='Messages per Day'
    )
   
 
    fig3 = px.bar(
        x=messages_per_day_of_week.index,
        y=messages_per_day_of_week.values,
        labels={'x': 'Day of the Week', 'y': 'Number of Messages'},
        title='Messages per Day of the Week'
    )

    # Messages per hour (bar plot with 12-hour AM/PM format)
    hour_labels = messages_per_hour.index.map(lambda x: f"{x % 12 or 12}{' AM' if x < 12 else ' PM'}")
    fig4 = px.bar(
        x=hour_labels,
        y=messages_per_hour.values,
        labels={'x': 'Hour (AM/PM)', 'y': 'Number of Messages'},
        title='Messages per Hour of the Day'
    )
    fig4.update_xaxes(tickangle=0)

    # Display plots in Streamlit
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)



def perform_date(whatsapp_df):
    # Convert timestamp strings to datetime objects for analysis
    whatsapp_df['timestamp'] = pd.to_datetime(whatsapp_df['timestamp'], format='%m/%d/%y, %I:%M:%S %p')

    # Extract date, time, day of week, and hour for further analysis
    whatsapp_df['date'] = whatsapp_df['timestamp'].dt.date
    whatsapp_df['time'] = whatsapp_df['timestamp'].dt.time
    whatsapp_df['day_of_week'] = whatsapp_df['timestamp'].dt.day_name()
    whatsapp_df['hour'] = whatsapp_df['timestamp'].dt.hour



from textblob import TextBlob

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


import matplotlib.pyplot as plt
import streamlit as st

def user_messages(whatsapp_df):
    
    # Counting messages per sender
    messages_per_sender = whatsapp_df['sender'].value_counts()

    # Plotting the number of messages sent by each user
    plt.figure(figsize=(10, 6))
    messages_per_sender.plot(kind='bar')
    plt.title('Number of Messages per User')
    plt.xlabel('User')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45)

    st.pyplot(plt)



import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import streamlit as st

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


import matplotlib.pyplot as plt

import streamlit as st


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


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
import streamlit as st

def forecast_message_trends(whatsapp_df):
    if 'date' not in whatsapp_df.columns:
        st.error("Date column not found in the data.")
        return

    # Prepare data
    message_counts = whatsapp_df.groupby('date').size().reset_index(name='count')

    # Ensure the 'date' column is a datetime object
    message_counts['date'] = pd.to_datetime(message_counts['date'])
    start_date = message_counts['date'].min()

    # Calculate the number of days since the start for each date
    message_counts['date_numeric'] = (message_counts['date'] - start_date).dt.days

    X = message_counts[['date_numeric']]
    y = message_counts['count']

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get user input for forecasting
    future_days = st.slider("Select number of days to forecast", 10, 60, 30)
    future_dates = pd.date_range(start=message_counts['date'].max() + pd.Timedelta(days=1), periods=future_days)

    # Calculate days since start for future dates
    future_dates_numeric = (future_dates - start_date).days

    # Make predictions
    future_predictions = model.predict(np.array(future_dates_numeric).reshape(-1, 1))

    # Create traces for historical and future trends
    trace_historical = go.Scatter(
        x=message_counts['date'],
        y=message_counts['count'],
        mode='lines+markers',
        name='Historical Message Counts',
        line=dict(color='blue')
    )

    trace_forecast = go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode='lines+markers',
        name='Predicted Message Counts',
        line=dict(color='red')
    )

    # Create the layout
    layout = go.Layout(
        title='Predicted Future Message Trends',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Number of Messages'),
        legend=dict(x=0, y=1),
        hovermode='x'
    )

    # Create the figure and plot it
    fig = go.Figure(data=[trace_historical, trace_forecast], layout=layout)
    st.plotly_chart(fig, use_container_width=True)



import streamlit as st


def display_alerts(whatsapp_df):
    # Define your alert conditions
    keywords = ['help', 'important', 'ASAP']
    sentiment_threshold = 0.7  # example threshold for positive sentiment

    # Check for keyword alerts and high sentiment messages
    alerts = []
    for index, row in whatsapp_df.iterrows():
        # Check for keyword alerts
        if any(keyword in row['message'].lower() for keyword in keywords):
            alerts.append((row['timestamp'], 'keyword', row['message']))

        # Check for sentiment alerts
        if row.get('polarity') and row['polarity'] > sentiment_threshold:
            alerts.append((row['timestamp'], 'sentiment', row['message']))

    # Display alerts
    if alerts:
        for alert in alerts:
            if alert[1] == 'keyword':
                # Display keyword alerts in one color (e.g., blue)
                st.markdown(f"<span style='color: blue;'>Alert at {alert[0]} due to {alert[1]}: {alert[2]}</span>", unsafe_allow_html=True)
            elif alert[1] == 'sentiment':
                # Display sentiment alerts in another color (e.g., green)
                st.markdown(f"<span style='color: green;'>Alert at {alert[0]} due to {alert[1]}: {alert[2]}</span>", unsafe_allow_html=True)
    else:
        st.write("No alerts based on the given conditions.")


from transformers import pipeline
import streamlit as st

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


from transformers import pipeline
import streamlit as st

def transformers_ner_analysis(whatsapp_df):
    # Load NER pipeline
    ner_pipeline = pipeline("ner", grouped_entities=True)

    # Display a selectbox to choose a message
    message_index = st.selectbox("Select a message index for NER", whatsapp_df.index)
    example_message = whatsapp_df.loc[message_index, 'message']

    # Perform NER
    if st.button("Perform NER"):
        result = ner_pipeline(example_message)
        st.write(f"Message: {example_message}")
        st.write("Named Entities:")
        for entity in result:
            st.write(f"{entity['entity_group']} ({entity['score']:.2f}): {entity['word']}")


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


from transformers import pipeline
import streamlit as st

def transformers_text_generation():
    # Load text generation pipeline
    generator = pipeline("text-generation", model="gpt2")

    # User input for starting text
    starting_text = st.text_input("Enter starting text for generation", "Let's plan a meetup")

    # Generate text
    if st.button("Generate Text"):
        with st.spinner("Generating..."):
            generated = generator(starting_text, max_length=50, num_return_sequences=1)
            st.write("Generated Text:")
            st.write(generated[0]['generated_text'])


import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def message_frequency(whatsapp_df):
    # Ensure the 'timestamp' column is in datetime format
    whatsapp_df['timestamp'] = pd.to_datetime(whatsapp_df['timestamp'])

    # Active Time Analysis: Count messages per hour of the day
    messages_per_hour = whatsapp_df['timestamp'].dt.hour.value_counts().sort_index()

    # Convert 24-hour format to 12-hour format (AM/PM) without list comprehension
    hours_12hr = []
    for h in range(24):
        if h == 0:
            hours_12hr.append("12 AM")
        elif h < 12:
            hours_12hr.append(f"{h} AM")
        elif h == 12:
            hours_12hr.append("12 PM")
        else:
            hours_12hr.append(f"{h - 12} PM")

    # Create an interactive bar chart for message frequency by hour
    fig1 = px.bar(
        x=messages_per_hour.index,
        y=messages_per_hour.values,
        labels={'x': 'Hour of Day', 'y': 'Number of Messages'},
        title='Message Frequency by Hour of Day',
        text=messages_per_hour.values
    )
    fig1.update_traces(textposition='outside')

    # Update x-axis to show 12-hour format with AM/PM
    fig1.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, 24)),
            ticktext=hours_12hr
        )
    )

    # Response Time Analysis: Calculate time differences between messages
    whatsapp_df['response_time'] = whatsapp_df['timestamp'].diff()

    # Convert time differences to minutes
    whatsapp_df['response_time_minutes'] = whatsapp_df['response_time'].dt.total_seconds() / 60

    # Calculate average response time
    average_response_time = whatsapp_df['response_time_minutes'].mean()

    # Display in Streamlit
    st.plotly_chart(fig1, use_container_width=True)
    st.write(f"Average response time: {average_response_time:.2f} minutes")

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


import streamlit as st
from collections import Counter
import re

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
    

def show_one_word_messages_count_top_users(whatsapp_df):
    
    whatsapp_df['is_one_word'] = whatsapp_df['word_count'] == 1
    one_word_counts = whatsapp_df[whatsapp_df['is_one_word']].groupby('sender').size().sort_values(ascending=False).head(5)
    st.subheader("One-Word Messages by Top 5 Users")
    st.bar_chart(one_word_counts)


import emoji

def show_emoji_usage_top_users(whatsapp_df):
    whatsapp_df = whatsapp_df[~whatsapp_df['sender'].str.contains('Messages and calls are end-to-end encrypted', na=False)]
    whatsapp_df = whatsapp_df[whatsapp_df['message'] != ""]
    whatsapp_df['emoji_count'] = whatsapp_df['message'].apply(lambda x: len([char for char in x if char in emoji.EMOJI_DATA]))
    max_emojis = whatsapp_df.groupby('sender')['emoji_count'].sum().sort_values(ascending=False).head(5)
    st.subheader("Emoji Usage by Top 5 Users")
    st.bar_chart(max_emojis)

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



import random

import streamlit as st

def mystery_user_challenge(data, sample_size=5):
    # Select a random subset of messages
    sample_messages = data.sample(sample_size)

    # Display the challenge
    st.markdown("### Mystery User Challenge")
    st.markdown("Guess who sent these messages:")

    # For storing user guesses
    guesses = []

    for index, row in sample_messages.iterrows():
        st.markdown(f"Message: '{row['message']}'")
        # Add a text input for each message for the user to enter their guess
        guess = st.text_input(f"Guess for message {index+1}", key=f"guess_{index}")
        guesses.append(guess)

    # Check if all guesses are made
    if all(guesses):
        st.markdown("### Your Guesses")
        for i, guess in enumerate(guesses, 1):
            st.write(f"Guess for message {i}: {guess}")

# Usage
#mystery_user_challenge(data)



from collections import Counter
import random

def chat_wordle(whatsapp_df):
    common_words = [word for word, count in Counter(" ".join(whatsapp_df['message']).split()).items() if count > 5]
    secret_word = random.choice(common_words)

    st.subheader("Chat Wordle")
    st.write("Guess the common word used in the chat:")
    user_guess = st.text_input("Enter your guess:")

    if user_guess:
        if user_guess.lower() == secret_word.lower():
            st.success("Correct! You guessed the word!")
        else:
            st.error("Wrong guess. Try again!")

from textblob import TextBlob




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

def report_generation(whatsapp_df):
    pass



def display_big_bold_centered_text(text, fontsize=None):
    # Set default font size if not provided
    if fontsize is None:
        fontsize = 30  # Default size

    st.markdown(f"""
        <div style="text-align: center; font-size: {fontsize}px; font-weight: bold;">
            {text}
        </div>
        """, unsafe_allow_html=True)





senti_text = """The visualizations for the sentiment analysis of your WhatsApp chat data provide insights into the emotional tone of the group conversations:

Distribution of Sentiment Polarity:

This histogram shows the frequency distribution of polarity scores across all messages. Polarity scores range from -1 (very negative) to +1 (very positive), with scores
around 0 indicating neutral sentiment. The shape of the distribution can give an idea of the overall positivity or negativity of the group's conversations.

Average Sentiment Polarity per Day:

This line plot shows the average sentiment polarity for each day. Fluctuations in this plot indicate changes in the group's overall mood on a day-to-day basis.
Peaks (high values) suggest days with more positive conversations, while troughs (low values) indicate days with more negative sentiments. These visualizations are
essential for understanding the emotional dynamics of the group over time and can be particularly insightful when correlated with specific events or topics
discussed in the group."""


topic_text = """This code performs the following steps:

Preprocesses the text data by tokenizing, converting to lowercase, removing non-alphabetic tokens, and filtering out stopwords. Creates a dictionary and
corpus needed for LDA topic modeling. Runs the LDA model to discover topics in the chat data. Displays the top words associated with each topic.
The output will be a set of topics, each represented by a set of words that are most significant to that topic. This can help you understand the main
themes of discussion in your WhatsApp group chat."""


emoji_text ="""Most Common Emojis: The list and bar chart of the most common emojis give a quick insight into the prevalent moods and reactions in the
group chat. For example, a preponderance of laughter or smiley emojis might suggest a generally light-hearted and positive group atmosphere.

Emoji Usage Patterns: By examining which emojis are most frequently used, you can infer the group's general mood and preferences. For instance:

Frequent use of hearts and smiley faces might indicate a friendly and positive interaction style. Use of surprise or shock emojis could imply
frequent sharing of surprising or unexpected news. Contextual Analysis: For a deeper understanding, consider the context in which these emojis are
used. This could involve analyzing the text surrounding the emoji usage to interpret the sentiments more accurately."""




forecasting_text = """Historical Message Counts (Blue Line): This represents the actual number of messages sent in the group for each time point in your
historical data. The spikes indicate days with a high number of messages, which could be due to specific events or conversations that engaged many members of the group.

Predicted Message Counts (Red Line): This shows the predicted number of messages based on the linear regression model. It appears as a flat line because
a simple linear regression model doesn't capture the variability or seasonality in the data; it only predicts the average trend.

Here's how to interpret the graph:

The blue line's spikes and troughs represent the natural variability in how many messages are sent each day. Some days are busier than others. The red line's
flatness indicates that the linear regression model predicts that, on average, the future will continue with the same average message count as the historical mean.
It does not predict the ups and downs because it's not a time-series model that captures patterns over time. For more accurate forecasting, especially for
time series data like this, you might consider using models that can account for seasonality, trends, and irregular cycles, such as ARIMA, SARIMA, or even LSTM networks
for deep learning approaches.

It's also worth noting that the linear model will not capture any potential future events that could cause spikes in messaging - it assumes the future will be like
the past, on average."""


word_cloud_text = """How to Interpret the Output: Word Clouds:

The size of each word in the word cloud indicates its frequency in the chat. Larger words were mentioned more frequently, highlighting the key themes or
subjects that dominate the group's conversations. It gives a quick visual representation of the most discussed topics. Trending Topics (Not included in the
code but typically involves Topic Modeling):

Trending topics analysis would identify the main themes or topics in the chat over time. Each topic would be represented by a set of words that frequently
occur together. By analyzing how the prominence of these topics changes over time, you can understand shifts in the group's focus or interest."""



# Custom CSS to adjust sidebar content positioning
st.markdown(
    """
    <style>
       
        section[data-testid="stSidebar"] {
            top: 0;
            left: 0;
            height: 100vh;
            background-color: black;
            
        }

        .main .block-container {
            padding-top: 30px !important;
            
        }
    </style>
    """,

    unsafe_allow_html=True
)

def main():
    # Display the sidebar at the top
    with st.sidebar:
        st.image('assets/whatsapp_logo.png',width=150)  # Display a logo
        st.write("## Navigation")
        analysis_option = st.selectbox("Choose the Analysis you want to perform",
                                    ["Home", "About the App","Show Data", "EDA", "Sentiment Analysis", "User Analysis",
                                        "Topic Analysis", "Emojis and Words Analysis", "Forecasting", "Alert",
                                        "Funny Analysis", "Transformers-Sentiment Analysis", "NER", "Summarization",
                                        "Text Generation", "Message Frequency", "Challenge", "Wordcloud"])

    
    if analysis_option == "Home":
            home.fyp()
    elif analysis_option == "About the App":
            
            
            #display_big_bold_centered_text("About the App")
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            
            st.write("""
            # 📊 WhatsApp Chat Analysis App  
            Welcome to the **WhatsApp Chat Analysis App**! This tool provides deep insights into your WhatsApp conversations through various analytical techniques.  

            ## 🔍 Features  
            - **📈 Exploratory Data Analysis (EDA):** Get a statistical overview of your chat data with insightful visualizations.  
            - **😊 Sentiment Analysis:** Determine the emotional tone of messages (positive, negative, neutral).  
            - **👥 User Analysis:** Explore user activity and participation in chats.  
            - **🗣️ Topic Analysis:** Identify common discussion topics in your chat.  
            - **🎭 Emojis & Word Usage:** Explore frequently used emojis and words.  
            - **⏳ Message Frequency:** Analyze chat activity trends over time.  
            - **📊 Forecasting:** Predict future message activity patterns.  
            - **🚨 Alerts:** Get notifications based on specific keywords or sentiments.  
            - **😂 Funny Analysis:** Identify humorous or engaging messages.  
            - **🤖 Transformers-Based Analysis:** Leverage AI for **Sentiment Analysis** and **Named Entity Recognition (NER)**.  
            - **📃 Summarization:** Generate concise summaries of lengthy conversations.  
            - **✍️ AI Text Generation:** Create AI-powered chat responses.  
            - **💡 Challenge Mode:** Discover interesting trends and patterns in your chat.  
            - **☁️ Word Cloud:** Visualize frequently used words in a creative word cloud.  

            🔹 **Uncover hidden insights from your WhatsApp chats and explore your conversations like never before! 🚀**
            """)


    # Create a placeholder for future outputs
    output_placeholder = st.empty()
    data =  load_and_parse_data.load_data()
    if data is not None:
        
        if analysis_option == "EDA":
            output_placeholder.empty()  # Clear previous output
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text("Exploratory Data Analysis (EDA) ",40)
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            perform_eda(data)
            pass

        elif analysis_option == "User Analysis":
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            
            
            display_big_bold_centered_text("""Detailed User Analysis""", 40)
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            show_most_frequent_words_by_users(data)
            show_word_count_top_users(data)
           # show_one_word_messages_top_users(data)
            show_emoji_usage_top_users(data)

        elif analysis_option == "Funny Analysis":
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")

            display_big_bold_centered_text("Funny Analysis",40)
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            most_active_time(data)
            laugh_counter(data)
            most_used_emojis(data)
            mood_meter(data)

        elif analysis_option == "Challenge":
            display_big_bold_centered_text("Challenge")
            mystery_user_challenge(data)
            chat_wordle(data)

        elif analysis_option == "Sentiment Analysis":
            output_placeholder.empty()  # Clear previous output
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text("Sentiment Analysis", 40)
            perform_date(data)
            analyzed_data = perform_sentiment_analysis(data)
            visualize_sentiment_analysis(analyzed_data)
            st.markdown(senti_text)
            pass

        elif analysis_option == "Topic Analysis":
            output_placeholder.empty()  # Clear previous output
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text("Topic Analysis",40)
            num_topics = st.slider("Select number of topics", 3, 10, 5)
            lda_model = perform_topic_modeling(data, num_topics=num_topics)
            visualize_topics(lda_model)
            st.markdown(topic_text)
            pass

        elif analysis_option == "Show Messages per User":
            output_placeholder.empty()  # Clear previous output
            display_big_bold_centered_text("Show Messages per User")
            user_messages(data)

        elif analysis_option == "Emojis and Words Analysis":
            output_placeholder.empty()  # Clear previous output
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text("Emojis and Words Analysis",40)
            
            processed_word_freq = preprocess_and_extract_words(data)
            emoji_freq = extract_and_count_emojis(data)
            visualize_words_and_emojis(processed_word_freq, emoji_freq)
            st.markdown(emoji_text)

        elif analysis_option == "Forecasting":
            output_placeholder.empty()  # Clear previous output
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text("Forecasting",40)
            perform_date(data)
            forecast_message_trends(data)
            st.markdown(forecasting_text)

        elif analysis_option == "Alert":
            output_placeholder.empty()  # Clear previous output
            display_big_bold_centered_text("Alerts",40)
            display_alerts(data)

        elif analysis_option == "Transformers-Sentiment Analysis":
            output_placeholder.empty()  # Clear previous output
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text("Transformers-Sentiment Analysis",40)
            display_big_bold_centered_text(" ")
            transformers_sentiment_analysis(data)

        elif analysis_option == "NER":
            output_placeholder.empty()  # Clear previous output
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text("Named Entity Recognition (NER)",40)
            display_big_bold_centered_text(" ")
            transformers_ner_analysis(data)

        elif analysis_option == "Summarization":
            output_placeholder.empty()  # Clear previous output
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text("Summarization",40)
            display_big_bold_centered_text(" ")
            transformers_text_summarization(data)

        elif analysis_option == "Text Generation":
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            output_placeholder.empty()  # Clear previous output
            display_big_bold_centered_text("Text Generation",40)
            display_big_bold_centered_text(" ")
            transformers_text_generation()

        elif analysis_option == "Message Frequency":
            output_placeholder.empty()  # Clear previous output
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text("Message Frequency",40)
            display_big_bold_centered_text(" ")
            message_frequency(data)

        elif analysis_option == "Wordcloud":
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            output_placeholder.empty()  # Clear previous output
            display_big_bold_centered_text("Wordcloud",40)
            display_big_bold_centered_text(" ")
            generate_wordcloud(data)
            display_big_bold_centered_text(" ")
            st.markdown(word_cloud_text)

        elif analysis_option == "Show Data":
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text(" ")
            display_big_bold_centered_text("Display the dataframe", 40)
            display_big_bold_centered_text(" ")
            st.dataframe(data.head(500))

                        

if __name__ == "__main__":
    main()

