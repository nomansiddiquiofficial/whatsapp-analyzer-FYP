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


