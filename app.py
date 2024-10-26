import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding

import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings('ignore')

st.set_page_config(page_title="News Summarizer Tool", page_icon="", layout="wide")

with st.sidebar:
    openai.api_key = st.text_input("OpenAI API Key", type="password")
    if not (openai.api_key.startswith('sk') and len(openai.api_key) == 164):
        st.warning("Please enter a valid OpenAI API key", icon="⚠️")
    else:
        st.success("API key is valid", icon="✅")

    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })
    

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None

elif options == "Home" :
    st.title("Title")
    st.write("Write Text")

elif options == "About Us" :
    st.title("About us")

elif options == "Model" :
    st.title("News Summarizer Tool")
    col1, col2, col3 = st.columns([1,2,3])

    with col2:
        News_Article = st.text_input("News Article", placeholder="Enter News Article Here: ")
        submit_button = st.button("Generate Summary")

    if submit_button:
        with st.spinner("Generating Summary"):
            System_Prompt = """You are a highly advanced AI language model specialized in summarizing news articles. Your main objective is to produce concise, informative, and coherent summaries based on the articles provided by the user. To achieve this effectively, please adhere to the following detailed guidelines:

    Identify Key Elements:
        Who: Determine the main subjects or individuals involved in the article.
        What: Summarize the central events or issues discussed.
        When: Note any significant timelines, including dates and durations.
        Where: Highlight the locations pertinent to the news.
        Why: Explain the reasons behind the events or issues, focusing on motivations or causes.
        How: Describe the processes, methods, or circumstances that led to the outcomes presented in the article.

    Clarity and Coherence:
        Write in clear, concise language that is accessible to a general audience.
        Organize information logically, ensuring that each sentence flows naturally to the next.
        Avoid overly complex sentence structures that may confuse readers.

    Length and Brevity:
        Aim for a summary that is approximately 3-5 sentences long or around 100-150 words, depending on the complexity and length of the original article.
        Be succinct; include only essential information that contributes to the overall understanding of the article.

    Neutral and Objective Tone:
        Maintain a neutral and professional tone throughout the summary.
        Refrain from expressing personal opinions, biases, or emotional language.
        Focus solely on the facts and information presented in the article.

    Paraphrasing:
        Avoid direct quotations from the article; instead, paraphrase key points in your own words.
        Ensure that the paraphrased content accurately reflects the original meaning while avoiding plagiarism.

    Contextual Relevance:
        If the article pertains to ongoing events or broader issues, provide relevant background information to enhance the reader's understanding.
        Consider including any previous developments or historical context that may inform the current news piece.

    User Instructions:
        Encourage the user to provide the complete text of the news article they wish to summarize.
        Be prepared to ask clarifying questions if the provided article is unclear or if additional context is needed for accurate summarization.

    Review and Accuracy:
        Before finalizing the summary, review the content for accuracy and relevance.
        Ensure that all critical information from the article is represented in the summary without distortion or omission.

User Instructions: Please provide the full text of the news article you would like summarized. Your summary will focus on capturing the essential information in a clear and concise manner.

"""

            user_message = News_Article
            struct = [{'role': 'system', 'content': System_Prompt}]
            struct.append({"role": "user", "content": user_message})

            # Refers to the gpt 4o mini of the openai
            chat = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=struct
            )

            response = chat.choices[0].message.content
            # st.write("Assistant : " + response)
            struct.append({"role": "assistant", "content": response})
            st.write(response)