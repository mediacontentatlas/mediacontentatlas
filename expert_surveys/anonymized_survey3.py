import streamlit as st
import streamlit_survey as ss
import pandas as pd
from datetime import datetime
import json
import torch
from itertools import islice
import os

# Initialize the survey with a title
survey = ss.StreamlitSurvey("Survey Example - Method Evaluation, Task: Image Retrieval")

# Set Streamlit page configurations
st.set_page_config(
    page_title="Evaluate Content-Based Image Retrieval", 
    page_icon="🤔", 
    layout="wide"
)

# Cache function to load the dataset
@st.cache_data
def load_dataframe(filepath):
    return pd.read_csv(filepath, low_memory=False)

# Load dataset (Ensure to replace with your file path)
df = load_dataframe("/path/to/your/dataset.csv")

# Directory where images are stored (Modify as needed)
image_dir = "/path/to/your/image/directory"

def load_responses_from_temp_file():
    """
    Load saved responses from a temporary file if it exists.
    """
    temp_filename = "survey_part3_responses_temp.json"
    if os.path.exists(temp_filename):
        with open(temp_filename, 'r') as f:
            st.session_state['responses'] = json.load(f)
        print(f"Progress loaded from {temp_filename}")
    else:
        print("No saved progress found. Starting a new survey.")

def save_responses():
    """
    Save responses to session state.
    """
    st.session_state['responses'] = {key: st.session_state.get(key, "") for key in st.session_state}

def save_responses_to_file():
    """
    Save the current responses to a JSON file with a timestamp.
    """
    save_responses()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f'survey_responses_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(st.session_state['responses'], f)
    st.success(f"Your responses have been saved to '{json_filename}'.")

def save_responses_to_temp_file():
    """
    Save responses to a temporary file for progress tracking.
    """
    save_responses()
    with open("survey_part3_responses_temp.json", 'w') as f:
        json.dump(st.session_state['responses'], f)

@st.cache_data
def retrieve_images(query, top_k, model):
    """
    Retrieve top-k images based on a query using precomputed similarity scores.
    """
    json_file = "/path/to/retrieval_results.json"
    
    with open(json_file, 'r') as f:
        data = json.load(f)
        if model in data and query in data[model]:
            filenames_similarities = data[model][query]
        else:
            st.error(f"Model '{model}' or query '{query}' not found in the data.")
            return []
    
    top_k_similar_images = dict(islice(filenames_similarities.items(), top_k))
    triples_list = [
        (filename, float(metrics.get('similarity', 0)), df.loc[df['Filename'] == filename]['Output'].values[0])
        for filename, metrics in top_k_similar_images.items()
    ]
    return triples_list

@st.cache_data
def initialize_session_state():
    """
    Initialize session state variables and load responses if available.
    """
    queries = ["An image with " + i + " content" for i in ["cat", "nature", "food", "Trump", "Biden","alcohol", "blood", "guns","Coca Cola","depression", "violent", "political", "health", "therapy", "substance abuse","bullying","shopping","gambling","finance","influencer"]]
    models = ["clip", "llava"]
    
    load_responses_from_temp_file()
    
    if 'retrieval_results' not in st.session_state:
        st.session_state['retrieval_results'] = {
            query.replace(' ', '_'): {model: retrieve_images(query, 5, model) for model in models}
            for query in queries
        }

    if 'responses' not in st.session_state:
        st.session_state['responses'] = {}

initialize_session_state()

# Define the number of survey pages
total_pages = len(st.session_state['retrieval_results']) + 2
pages = survey.pages(total_pages + 1, on_submit=save_responses_to_file)

# Survey content
with pages:
    if pages.current == 0:
        st.header("Welcome to the Evaluation Survey")
        st.write("In this survey, you'll evaluate a content-based image retrieval system.") #add more details
        st.write("Click 'Next' to start the evaluation.")

      #make model a and b, and put the concretes first and then after each page ask for pattern feedback
    elif 1 <= pages.current <total_pages-1:
        page_responses = {}
        for query_key, models_dict in st.session_state['retrieval_results'].items():
            if pages.current == list(st.session_state['retrieval_results'].keys()).index(query_key) + 1:
                query = query_key.replace('_', ' ')
                st.header(f"Image Retrieval for Query: {query}")
                for model, retrieval_results in models_dict.items():
                    if model == "clip":
                        model_show = "A"
                    else:
                        model_show = "B"

                    st.write(f"## Retrieval Results of Model: {model_show}")
                    for i, (image_path, similarity_score, description) in enumerate(retrieval_results):
                        st.write(f"**Image {i+1}**")
                        st.image(f"{imagedir}/{image_path}", caption=f"Participant: {image_path.split('@')[0]},Similarity Score: {similarity_score:.2f}, Description: {description}", width=400)

                        slider_value = st.select_slider(
                            f"How **relevant** is Image {i+1} above to the query **'{query}'** (Model: {model_show})?",
                            options=["Highly Irrelevant","Irrelevant","Slightly Irrelevant", "No Answer", "Slightly Relevant", "Relevant", "Highly Relevant"],
                            value=st.session_state['responses'][f"{query_key}_{model}_retrievalslider_{i}_page_{pages.current}"]
                        )

                        comment_value = st.text_area(
                            f"Please provide any additional comments for your rationale for your answers about Image {i+1} (Model: {model_show}):",
                            value=st.session_state['responses'][f"{query_key}_{model}_retrievalcomment_{i}_page_{pages.current}"]
                        )
                        page_responses[f"{query_key}_{model}_retrievalslider_{i}_page_{pages.current}"] = slider_value
                        page_responses[f"{query_key}_{model}_retrievalcomment_{i}_page_{pages.current}"] = comment_value
                    st.write("----------------------------------------")
                st.write("#### Based on the images above retrieved by Models A and B:")
                pattern_feedback=st.text_area("What patterns or trends do you notice in the image retrieval results across different models for the same query?", 
                             value=st.session_state['responses'][f"pattern_feedback_{query_key}_page_{pages.current}"])
                page_responses[f"pattern_feedback_{query_key}_page_{pages.current}"] = pattern_feedback
        st.session_state['responses'].update(page_responses)
        save_responses_to_temp_file()

    elif pages.current == total_pages-1:
        st.header("Free Form Image Retrieval")
        st.write("We are developing a live image retrieval system that can retrieve images based on any user query. When it is ready, what queries would you like to test?")
        st.write("Please provide a list of queries separated by commas (e.g., 'dog, a photo of a church, a screenshot related to climate change, an image containing health-related information').")
        free_form_queries=st.text_area(
            "Queries for Image Retrieval",
            value=st.session_state['responses']["free_form_queries"]
        )
        st.session_state['responses']["free_form_queries"] = free_form_queries
        save_responses_to_temp_file()


    elif pages.current == total_pages:
        st.header("Final Comments")
        final_comments=st.text_area(
            "Any additional comments or feedback you would like to provide about the image retrieval results?",
            value=st.session_state['responses']["final_comments"]
        )
        st.session_state['responses']["final_comments"] = final_comments
        save_responses_to_temp_file()

        st.write("#### Thank you for your participation! Please click 'Submit' to complete the survey.")