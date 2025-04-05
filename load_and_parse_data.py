import streamlit as st 
import re
import pandas as pd

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
