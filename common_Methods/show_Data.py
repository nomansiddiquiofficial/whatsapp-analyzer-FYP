import streamlit as st
def show_Data(data):
    st.dataframe(data.head(500))
