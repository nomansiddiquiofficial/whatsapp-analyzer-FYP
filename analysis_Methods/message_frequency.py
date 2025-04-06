import pandas as pd
import plotly.express as px
import streamlit as st

def message_frequency(whatsapp_df):
    # Ensure the 'timestamp' column is in datetime format
    whatsapp_df['timestamp'] = pd.to_datetime(whatsapp_df['timestamp'])

    # --- Active Time Analysis ---
    messages_per_hour = whatsapp_df['timestamp'].dt.hour.value_counts().sort_index()

    # Convert 24-hour format to 12-hour format (AM/PM)
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

    # Create interactive bar chart
    fig1 = px.bar(
        x=messages_per_hour.index,
        y=messages_per_hour.values,
        labels={'x': 'Hour of Day', 'y': 'Number of Messages'},
        title='Message Frequency by Hour of Day',
        text=messages_per_hour.values
    )
    fig1.update_traces(textposition='outside')

    fig1.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(24)),
            ticktext=hours_12hr
        )
    )

    # --- Response Time Analysis ---
    whatsapp_df['response_time'] = whatsapp_df['timestamp'].diff()
    whatsapp_df['response_time_minutes'] = whatsapp_df['response_time'].dt.total_seconds() / 60

    # Remove NaN values and outliers (optional: >12 hours)
    valid_response_times = whatsapp_df['response_time_minutes'].dropna()
    valid_response_times = valid_response_times[valid_response_times <= 720]  # 12 hours

    # Calculate average response time
    average_response_time = valid_response_times.mean()

    # --- Display in Streamlit ---
    st.plotly_chart(fig1, use_container_width=True)
    st.write(f"Average response time (excluding outliers): {average_response_time:.2f} minutes")
