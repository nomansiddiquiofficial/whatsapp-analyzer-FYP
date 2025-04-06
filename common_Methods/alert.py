import streamlit as st
def alert(whatsapp_df):
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


