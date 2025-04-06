from common_Methods.display_big_Centered import display_big_bold_centered_text
import streamlit as st
import plotly.express as px
def perform_eda(whatsapp_df):
# Filter out system messages like encryption notice
    whatsapp_df = whatsapp_df[~whatsapp_df['sender'].str.contains('Messages and calls are end-to-end encrypted', na=False)]

    # Extract date, time, day of week, and hour for further analysis
    whatsapp_df['date'] = whatsapp_df['timestamp'].dt.date
    whatsapp_df['time'] = whatsapp_df['timestamp'].dt.time
    whatsapp_df['day_of_week'] = whatsapp_df['timestamp'].dt.day_name()
    whatsapp_df['hour'] = whatsapp_df['timestamp'].dt.hour

    # Metrics for card
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
