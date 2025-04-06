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
