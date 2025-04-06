import streamlit as st
def display_big_bold_centered_text(text, fontsize=None):

    # Set default font size if not provided
    if fontsize is None:
        fontsize = 30  # Default size

    st.markdown(f"""
        <div style="text-align: center; font-size: {fontsize}px; font-weight: bold;">
            {text}
        </div>
        """, unsafe_allow_html=True)
