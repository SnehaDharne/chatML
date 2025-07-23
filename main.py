import os
from dotenv import load_dotenv
import pandas as pd
from google import generativeai as genai
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-pro")

import streamlit as st
from model_executor import run_model_from_json
from load_context import load_context
import google.generativeai as genai
import os, re, json



def clean_json_response(response):
    """
    Strip ``` blocks or other formatting Gemini might return.
    """
    # remove markdown-style triple backticks
    clean = re.sub(r"```(?:json)?", "", response).strip("` \n")
    return clean


def rag(chat_history, question, ml_model, model):
    context = load_context(model=ml_model, question=question)

    prompt = f"""
You are a helpful ML assistant.

Your job is to:
1. Answer user questions using the provided context and chat history.
2. If the user wants to *train or build a model*, return ONLY a dictionary with the following format:

- The dictionary key must be **"-1"**
- Its value must be a **valid JSON object** with the following structure:

Example:
{{
  "-1": {{
    "filename": "uploaded_data.csv",
    "model_name": "decision_tree", // can be svm, logistic_regression or decision_tree
    "target_variable": "Class",
    "split": 0.2,
    "param": {{
      "max_depth": 4
    }}
  }}
}}

⚠️ Only output this dictionary — no extra text, no commentary — when model training is requested.

Otherwise, answer normally based on the context and chat history.

---
Chat History:
{chat_history}

---
Context Docs:
{context}

---
User Question:
{question}
"""

    gemini_response = model.generate_content(prompt)
    return gemini_response.text.strip()


import json

def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.set_page_config(page_title="ChatML Assistant (Gemini)", layout="centered")
    st.title("ChatML powered by Gemini")
    
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    uploaded_filename = None

    if uploaded_file:
        uploaded_filename = "uploaded_data.csv"
        with open(uploaded_filename, "wb") as f:
            f.write(uploaded_file.read())

        df = pd.read_csv(uploaded_filename)
        st.success(f"Uploaded {uploaded_filename} with shape {df.shape}")
        st.dataframe(df.head())
    user_query = st.chat_input("Ask something like 'Use decision tree on the uploaded_data'")
    ml_model = st.selectbox(label='models', options=['svm', 'lr', 'dt'])
    
    for user_msg, assistant_msg in st.session_state.chat_history:
        st.chat_message("user").write(user_msg)
        st.chat_message("assistant").write(assistant_msg)
    if user_query:
        
        response = rag(st.session_state.chat_history, user_query, ml_model, model)
        cleaned_response = clean_json_response(response)
        
        try:
            parsed = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            # st.error("Could not parse response as JSON.")
            # st.exception(e)
            parsed = None

        if isinstance(parsed, dict) and "-1" in parsed:
            st.warning("Model build triggered based on your inputs...")
            try:
                metrics = run_model_from_json(parsed["-1"])
                st.success(f"Model trained! Accuracy: {metrics['accuracy']:.4f}")
                st.json(metrics)
                st.session_state.chat_history.append((f'metrics after the run with {parsed["-1"]}', f'{st.json(metrics)}'))
            except Exception as e:
                st.error(f"Error during model training: {e}")
        else:
            st.chat_message("user").write(user_query)
            st.chat_message("assistant").write(response)

        st.session_state.chat_history.append((user_query, response))


if __name__ == "__main__":
    main()
