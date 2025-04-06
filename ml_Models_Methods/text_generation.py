
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


