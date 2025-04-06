from analysis_Methods.message_frequency import message_frequency
from analysis_Methods.emojis_and_words_anaylsis import preprocess_and_extract_words, extract_and_count_emojis, visualize_words_and_emojis
from analysis_Methods.eda import perform_eda, user_messages
from analysis_Methods.funny_analysis import laugh_counter, mood_meter, most_active_time, most_used_emojis
from analysis_Methods.user_anlaysis import show_emoji_usage_top_users, show_most_frequent_words_by_users, show_one_word_messages_count_top_users, show_word_count_top_users
from analysis_Methods.wordcloud import generate_wordcloud
from common_Methods.about import about_app
from common_Methods.alert import alert
from common_Methods.challenge import chat_wordle, mystery_user_challenge
from common_Methods.display_big_Centered import display_big_bold_centered_text
from common_Methods.show_Data import show_Data
import common_Methods.load_and_parse_data as load_and_parse_data
import common_Methods.home as home
import streamlit as st
import pandas as pd
import re
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from ml_Models_Methods.ner import transformers_ner_analysis
from ml_Models_Methods.perform_sentiment_analysis import perform_sentiment_analysis, visualize_sentiment_analysis
from ml_Models_Methods.Forcasting  import forecast_message_trends, perform_date
from ml_Models_Methods.text_generation import transformers_text_generation
from ml_Models_Methods.topic_analysis import perform_topic_modeling, visualize_topics
from ml_Models_Methods.transformer_sentiment import transformers_sentiment_analysis
from ml_Models_Methods.text_summarization import transformers_text_summarization














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
           about_app() 
            
            


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
            show_one_word_messages_count_top_users(data)
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
            alert(data)

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
            show_Data(data)
                        

if __name__ == "__main__":
    main()

