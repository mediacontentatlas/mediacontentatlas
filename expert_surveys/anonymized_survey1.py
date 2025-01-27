import streamlit as st
import streamlit_survey as ss
import random
import pandas as pd
from datetime import datetime
import json
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np
import torch
from torch import tensor
from itertools import islice
from scipy.spatial import distance_matrix
import textwrap
import plotly.express as px
import os

# Initialize Survey
survey = ss.StreamlitSurvey("Survey - Method Evaluation")
st.set_page_config(page_title="Evaluate MCA Content Analysis - PART 1", page_icon="🧐", layout="wide", initial_sidebar_state="auto", menu_items=None)


# Function to load responses from the temporary file if it exists
def load_responses_from_temp_file():
    temp_filename = "survey_part1_responses_temp.json"
    if os.path.exists(temp_filename):
        with open(temp_filename, 'r') as f:
            st.session_state['responses'] = json.load(f)
        print(f"Progress loaded from {temp_filename}")
    else:
        print("No saved progress found. Starting a new survey.")


def initialize_session_state():
    load_responses_from_temp_file()

    # Only initialize responses if they don't exist in session state
    if 'responses' not in st.session_state:
        st.session_state['responses'] = {}
    
    # Set defaults for each key only if they haven't been saved yet
    defaults = {
        'content_informativeness_slider_0': "Equally Informative",
        'descriptiveness_comparison_slider_0': "Equally Descriptive",
        'usefulness_comparison_slider_0': "Equally Useful",
        'comparison_comments_0': "",
        'likelihood_use_radio_4': "Neutral",
        'usefulness_text_area_4': "",
        'suggestions_text_area_4': ""
    }

    # Initialize default values for keys if they are not already set in session state
    for key, default_value in defaults.items():
        if key not in st.session_state['responses']:
            st.session_state['responses'][key] = default_value




initialize_session_state()

def save_responses():
    # Update the current responses without overwriting the existing ones
    st.session_state['responses'] = {
        **st.session_state['responses'],  # Keep all previously saved responses
        'content_informativeness_slider_0': st.session_state.get('content_informativeness_slider_0', st.session_state['responses']['content_informativeness_slider_0']),
        'descriptiveness_comparison_slider_0': st.session_state.get('descriptiveness_comparison_slider_0', st.session_state['responses']['descriptiveness_comparison_slider_0']),
        'usefulness_comparison_slider_0': st.session_state.get('usefulness_comparison_slider_0', st.session_state['responses']['usefulness_comparison_slider_0']),
        'comparison_comments_0': st.session_state.get('comparison_comments_0', st.session_state['responses']['comparison_comments_0']),
        
        'likelihood_use_radio_4': st.session_state.get('likelihood_use_radio_4', st.session_state['responses']['likelihood_use_radio_4']),
        'usefulness_text_area_4': st.session_state.get('usefulness_text_area_4', st.session_state['responses']['usefulness_text_area_4']),
        'suggestions_text_area_4': st.session_state.get('suggestions_text_area_4', st.session_state['responses']['suggestions_text_area_4'])
    }



# Function to save responses to a temporary file
def save_responses_to_temp_file():
    # First, update the session state responses to ensure we have the latest values
    save_responses()

    # Now, save the session state to the temp file
    temp_filename = "survey_part1_responses_temp.json"
    
    # Save responses to the temporary file
    with open(temp_filename, 'w') as f:
        json.dump(st.session_state['responses'], f)  # Save the current state of responses
    print(f"Your progress has been auto-saved to {temp_filename}")


# Function to finalize responses and rename the file
def finalize_responses():
    temp_filename = "survey_part1_responses_temp.json"
    final_filename = f"survey_part1_responses_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.rename(temp_filename, final_filename)
    st.success(f"Your survey is completed and saved as {final_filename}")



def finalize_responses_on_submit():
    save_responses()  # Ensure the latest responses are saved to temp
    finalize_responses()  # Final save and rename the file

# Function to save responses to a temporary file

@st.cache_data
def show_viz(html_file_path):
    with open(html_file_path, "r") as html_file:
        html_content = html_file.read()
        st.components.v1.html(html_content,height=1000,width=None)


# Define pages
pages = survey.pages(6, on_submit=finalize_responses_on_submit)

with pages:
    if pages.current == 0:
        st.header("Welcome to the Evaluation Survey for Content-based Analysis of Screenomes!")

        # Introduction
        st.write("""
        We're excited to have you here and value your expertise in studying media processes! 
         
        In this survey, you will help us evaluate a novel method for understanding the content people interact with on their screens. 
        Traditionally, content analysis focuses on app names (e.g., Instagram, Twitter, Google Chrome) and app categories 
        (e.g., social, communication, gaming), but this approach only offers a surface-level view of user behavior.""")
        ## Add more details about the method and the evaluation process 
       

    elif pages.current == 1:
        #interactive visualization tutorial
        st.header("Interactive Plot Tutorial")

        st.write("""
        Before you start the survey, let's get familiar with the interactive features of the scatter plots you'll be using.
        These plots allow you to explore and analyze data by interacting with the points and visualizing clusters. In this tutorial, we'll use a simplified version of the data you'll encounter in the main survey.
        """)

         # Tutorial instructions
        st.write("""
        **How to interact with the scatter plot:**
        There are many ways to interact with this plot, and you can select one or multiple to make the most out of your experience.
        - **Hover**: Hover over the points to see more information, including the app name, category, and a brief description.
        - **Toggle topics**: Click on a topic in the legend to show or hide points related to that topic. This can help you see the distribution of the topic's points in the plot.
        - **Zoom**: Double click, or use your trackpad(2 fingers up) to zoom into specific area. Use trackpad(2 fingers down) to zoom out.
        - **Annotations**: See the labels for each topic directly on the plot.
        - **Search**: Search for a specific metadata(a specific app or a keyword) using the search bar. This search is pretty basic,  works as a quick filter.

        Take a few moments to explore this interactive plot. Once you're comfortable, click "Next" to proceed to the actual survey.
        """)

        
        with open("/path/to/interactive/tutorial/html/", "r") as html_file:
                html_content = html_file.read()
                st.components.v1.html(html_content,height=1000,width=None)
        
       
        
    elif pages.current == 2:
        st.header("PART 1")
        st.write("""
            This part consists of several general tasks designed for you to evaluate the results of our method (content-based clustering) compared to the baseline through qualitative exploration.
            
            Here's what you can expect, please read it carefully and ask any questions you may have before proceeding:
                 
            PART 1
                 
            1. **Explore the Baseline:** First, you will see an interactive visualization of ~1.12 Million screenshots, colored by app name and category which are traditional classifications of screen use. You can select different coloring of the same plot to explore the baseline further.
            2. **Our Method - Content-Based Clustering:** Next, you will see the same visualization colored by inferred topics based on the content of the screenshots.
            3. **Compare Baseline with Our Method:** Last, you will answer the survey questions to assess the usefulness and informative value of content-based clustering in comparison with the baseline.
            
            We invite you to approach the task as you would in your own research, take your time to explore the results, and provide detailed feedback.

            Please proceed by clicking "Next". You can always click "Previous" to go back and change your answers. You may need to wait in between answers and pages for the responses to be saved and visualizations to load.
        """)
    elif pages.current == 3:
        st.header("PART 1")
        st.header("1. Explore the Baseline: App Name and App Category-Based Classification of Screens")
        st.write("#### - Below is a visualization of ~1.12 Million screenshots colored by the traditional classifications of screen use - App Name and Category.")
        st.write("#### - Please explore the interactive chart on full screen. Use your trackpad to zoom to explore clusters. Hover over points to see details. You can search apps, categories, keywords and participants using search bar.")
        st.write("#### - Evaluation Goal: Next, you will explore the results of content-based clustering and will be asked to assess the relevance and informative value of content-based image clusters in comparison with the baselines below.")

        visualization_option = st.radio(
            "Select the type of visualization you'd like to see:",
            ("App-Based Coloring", "App Category-Based Coloring", "Participant-Based Coloring")
        )
        


        # Open the corresponding HTML visualization based on the user's choice
        if visualization_option == "App-Based Coloring":
            html_file_path = "/path/to/app/based/viz/html/"
            show_viz(html_file_path)        
        elif visualization_option == "App Category-Based Coloring":
            html_file_path = "/path/to/appcategory/based/viz/html/"
            show_viz(html_file_path)
        elif visualization_option == "Participant-Based Coloring":
            html_file_path = "/path/to/participant/based/viz/html/"
            show_viz(html_file_path)
            

    elif pages.current == 4:
        st.header("2. Our Method: Content-Based Clustering + Screen Description + App Name and Category")
        st.write("### - In this section, you will first see an interactive visualization of clusters colored by inferred topics based on the screenshot content.")
        st.write("### - Please explore the interactive chart on full screen. Use the zoom tool to explore clusters. Hover over points to see details.")
        st.write("### - Evaluation Goal: In the survey next page, you will assess the usefulness and informative value of content-based clusters in comparison with the baseline. ")
        html_file_path_topic = "/path/to/topic/based/MCA/viz/html/"
        
        #See visualization on the next tab and answer the 
        show_viz(html_file_path_topic)

    elif pages.current == 5:
        st.write("### Now, take a step back and evaluate your experience as whole with this content clustering and visualization method, and answer the following questions.")
        survey.select_slider(
            "How **informative** do you find **content-based clustering** compared to **the baseline (app name and app category)**?",
            options = ["Much Less Informative","Less Informative","Slightly Less Informative","Equally Informative","Slightly More Informative","More Informative","Much More Informative"],
            value=st.session_state['responses']['content_informativeness_slider_0'],
            id="content_informativeness_slider_0",
            key="content_informativeness_slider_0",
            on_change=save_responses_to_temp_file
        )
        
        survey.select_slider(
            "How **descriptive** is **content-based clustering** compared to **the baseline (app name and app category)**?",
            options=["Much Less Descriptive", "Less Descriptive","Slightly Less Descriptive", "Equally Descriptive","Slightly More Descriptive", "More Descriptive", "Much More Descriptive"],
            value=st.session_state['responses']['descriptiveness_comparison_slider_0'],
            id="descriptiveness_comparison_slider_0",
            key="descriptiveness_comparison_slider_0",
            on_change=save_responses_to_temp_file
        )
        
        survey.select_slider(
            "How **useful** is **content-based clustering** for your research compared to **the baseline (app name and app category)**?",
            options=["Much Less Useful", "Less Useful","Slightly Less Useful", "Equally Useful","Slightly More Useful", "More Useful", "Much More Useful"],
            value=st.session_state['responses']['usefulness_comparison_slider_0'],
            id="usefulness_comparison_slider_0",
            key="usefulness_comparison_slider_0",
            on_change=save_responses_to_temp_file
        )
        survey.text_area(
            "Any additional comments comparing the two methods?:",
            value=st.session_state['responses']['comparison_comments_0'],
            id="comparison_comments_0",
            key="comparison_comments_0",
            on_change=save_responses_to_temp_file
        )

        # Likelihood of using the method for research
        survey.radio(
            "How likely are you to use this method for research in analyzing screen content?",
            options=["Very Unlikely", "Unlikely", "Neutral", "Likely", "Very Likely"],
            index=["Very Unlikely", "Unlikely", "Neutral", "Likely", "Very Likely"].index(st.session_state['responses']['likelihood_use_radio_4']),
            id="likelihood_use_radio_4",
            key="likelihood_use_radio_4",
            on_change=save_responses_to_temp_file
        )
                
        # Specific scenarios or use cases
        survey.text_area(
                "How would this method be useful for your research? Please describe specific scenarios or use cases where this method would be beneficial.",
                value=st.session_state['responses']['usefulness_text_area_4'],
                id="usefulness_text_area_4",
                key="usefulness_text_area_4",
                on_change=save_responses_to_temp_file
            )

        # Suggestions for improvement
        survey.text_area(
                "What are your suggestions for the applications of this method? Please provide any suggestions you have for improving the application of this method, including any features or functionalities you would like to see added and any potential challenges or limitations you foresee.",
                value=st.session_state['responses']['suggestions_text_area_4'],
                id="suggestions_text_area_4",
                key="suggestions_text_area_4",
                on_change=save_responses_to_temp_file
            )
        
        st.write("#### Thank you for your participation! Please click 'Submit' to complete the survey.")

