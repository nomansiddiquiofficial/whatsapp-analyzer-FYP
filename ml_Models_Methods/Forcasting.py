import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objs as go

def perform_date(whatsapp_df):
    # Convert timestamp strings to datetime objects for analysis
    whatsapp_df['timestamp'] = pd.to_datetime(whatsapp_df['timestamp'], format='%m/%d/%y, %I:%M:%S %p')

    # Extract date, time, day of week, and hour for further analysis
    whatsapp_df['date'] = whatsapp_df['timestamp'].dt.date
    whatsapp_df['time'] = whatsapp_df['timestamp'].dt.time
    whatsapp_df['day_of_week'] = whatsapp_df['timestamp'].dt.day_name()
    whatsapp_df['hour'] = whatsapp_df['timestamp'].dt.hour

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

