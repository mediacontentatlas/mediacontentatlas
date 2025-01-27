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
survey = ss.StreamlitSurvey("Survey Example - Method Evaluation, Task: Clustering")
st.set_page_config(page_title="Evaluate MCA Content Analysis - PART 2", page_icon="🧐", layout="wide", initial_sidebar_state="auto", menu_items=None)
imagedir="/path/to/your/image/directory" #change this to the path of the images


@st.cache_data
def load_dataframe(filepath):
    return pd.read_csv(filepath, low_memory=False)
df = load_dataframe("/path/to/your/dataset.csv")

@st.cache_data
def pick_random_clusters(df,preselected_filename, num_clusters=5, num_images=10, preselected=True):
    if preselected==True:
        preselected_df=pd.read_csv(preselected_filename)
        unique_topics = np.unique(preselected_df['detailed_topic_1_labels'])
        cluster_data = {}
        for topic in unique_topics:
            cluster_images = preselected_df[preselected_df['detailed_topic_1_labels'] == topic].sample(n=10, random_state=42)[['participant','app','app_category','Document','Filename','detailed_topic_1_labels']]
            cluster_data[topic] = cluster_images
    else:
        unique_topics = np.unique(df['detailed_topic_1_labels'])
        eligible_topics = unique_topics[unique_topics != "Mobile App User Interface"] #remove the "noise" label
        random_clusters = random.sample(list(eligible_topics), num_clusters)
        cluster_data = {}
        for topic in random_clusters:
            cluster_images = df[df['detailed_topic_1_labels'] == topic].sample(n=num_images, random_state=42)[['participant','app','app_category','Document','Filename','detailed_topic_1_labels']]
            cluster_data[topic] = cluster_images
    return cluster_data

@st.cache_data
def pick_grid_images(df,random_clusters_2):
    grid_images={}
    for cluster_key in random_clusters_2.keys():
        sampled_images=random_clusters_2[cluster_key]
        sampled_filenames=sampled_images['Filename'].tolist()
        all_images=df[df['detailed_topic_1_labels']==cluster_key] #change the column name
        #all_images=all_images[all_images['participant'].isin(participants)]
        #sample 25 or if there are less than 25 images, sample all
        if len(all_images)<25:
            grid_images[cluster_key]=all_images
        else:   
            grid_images[cluster_key]=all_images.sample(n=25, random_state=42)
    return grid_images

# Function to load responses from the temporary file if it exists
def load_responses_from_temp_file():
    temp_filename = "survey_part2_responses_temp.json"
    if os.path.exists(temp_filename):
        with open(temp_filename, 'r') as f:
            st.session_state['responses'] = json.load(f)
        print(f"Progress loaded from {temp_filename}")
    else:
        print("No saved progress found. Starting a new survey.")

def initialize_session_state():
    # Load responses from temp file first
    temp_filename = "survey_part2_responses_temp.json"
    if os.path.exists(temp_filename):
        with open(temp_filename, 'r') as f:
            st.session_state['responses'] = json.load(f)
        print(f"Loaded responses from {temp_filename}")
    else:
        st.session_state['responses'] = {}

    # Add defaults only for missing keys
    for cluster_key, cluster_images in st.session_state['random_clusters_1'].items():
        for i, row in cluster_images.iterrows():
            filename = row['Filename']
            if f"relevance_{filename}" not in st.session_state['responses']:
                st.session_state['responses'][f"relevance_{filename}"] = "No Answer"
            if f"description_{filename}" not in st.session_state['responses']:
                st.session_state['responses'][f"description_{filename}"] = "No Answer"
            if f"comment_{filename}" not in st.session_state['responses']:
                st.session_state['responses'][f"comment_{filename}"] = ""

    for cluster_key, cluster_images in st.session_state['random_clusters_2'].items():
        for i, row in cluster_images.iterrows():
            filename = row['Filename']
            if f"similarity_slider_{filename}" not in st.session_state['responses']:
                st.session_state['responses'][f"similarity_slider_{filename}"] = "No Answer"
            if f"similarity_comments_{filename}" not in st.session_state['responses']:
                st.session_state['responses'][f"similarity_comments_{filename}"] = ""

    # Ensure final comments have a default if not already loaded
    if "final_comments" not in st.session_state['responses']:
        st.session_state['responses']["final_comments"] = ""

    print("Session state initialized:", st.session_state['responses'])

def save_responses():

    # Update `responses` only for the keys that have been modified
    for cluster_key, cluster_images in st.session_state['random_clusters_1'].items():
        for i, row in cluster_images.iterrows():
            filename = row['Filename']
            if f"relevance_{filename}" in st.session_state:
                st.session_state['responses'][f"relevance_{filename}"] = st.session_state[f"relevance_{filename}"]
            if f"description_{filename}" in st.session_state:
                st.session_state['responses'][f"description_{filename}"] = st.session_state[f"description_{filename}"]
            if f"comment_{filename}" in st.session_state:
                st.session_state['responses'][f"comment_{filename}"] = st.session_state[f"comment_{filename}"]

    for cluster_key, cluster_images in st.session_state['random_clusters_2'].items():
        for i, row in cluster_images.iterrows():
            filename = row['Filename']
            if f"similarity_slider_{filename}" in st.session_state:
                st.session_state['responses'][f"similarity_slider_{filename}"] = st.session_state[f"similarity_slider_{filename}"]
            if f"similarity_comments_{filename}" in st.session_state:
                st.session_state['responses'][f"similarity_comments_{filename}"] = st.session_state[f"similarity_comments_{filename}"]

    # Final comments are updated separately
    if "final_comments" in st.session_state:
        st.session_state['responses']["final_comments"] = st.session_state["final_comments"]

def save_responses_to_temp_file():
    save_responses() 
    temp_filename = "survey_part2_responses_temp.json"

    # Save responses to the temporary file
    with open(temp_filename, 'w') as f:
        json.dump(st.session_state['responses'], f)  # Save the current state of responses
    print(f"Your progress has been auto-saved to {temp_filename}")

# Function to save responses to a final file and remove the temp file
def save_responses_to_file():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f'survey_part2_responses_final_{timestamp}.json'

    # Save responses to the final file
    with open(json_filename, 'w') as f:
        json.dump(st.session_state['responses'], f)
    st.success(f"Your responses have been recorded and saved to '{json_filename}'.")

    # Remove the temporary file after saving the final responses
    temp_filename = "survey_part2_responses_temp.json"
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        print(f"Temporary file '{temp_filename}' has been deleted.")

# Initialize random clusters if not already initialized
def initialize_random_clusters():
    if 'random_clusters_1' not in st.session_state:
        st.session_state['random_clusters_1'] = pick_random_clusters(df,"path/to/the/randomly/selected/clusters", num_clusters=5)

    if 'random_clusters_2' not in st.session_state:
        st.session_state['random_clusters_2'] = pick_random_clusters(df,"path/to/the/randomly/selected/clusters", num_clusters=5)

    if 'grid_images' not in st.session_state:
        st.session_state['grid_images'] = pick_grid_images(df, st.session_state['random_clusters_2'])

# Call this function to ensure clusters are initialized safely
initialize_random_clusters()

# Call this function to initialize session state responses
initialize_session_state()

#check the number of columns and the width of the images
def display_grid(grid_images):
    st.write("### Cluster Images")
    num_cols = 5  # 5 images per row
    rows = len(grid_images) // num_cols
    for row_idx in range(rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            img_idx = row_idx * num_cols + col_idx
            img_row = grid_images.iloc[img_idx]
            cols[col_idx].image(f"{imagedir}/{img_row['Filename']}", width=200)  # Show the grid with smaller images on the right


# Define pages
pages = survey.pages(13, on_submit=save_responses_to_file)
page_responses = {}
with pages:
    if pages.current == 0:
        st.header("Welcome to the Evaluation Survey for Content-based Analysis of Screen Data!")
        #add details about the survey
        st.write("Please click Next to start the survey. You can go back and change your answers at any time.")
        
    elif pages.current == 1:
        st.header("PART 2")
        st.write("""
            This part consists of several deep-dive tasks designed for you to evaluate the results of clustering and topic modeling features of our method.
            
            Here's what you can expect, please read it carefully and ask any questions you may have before proceeding:
                 
            PART 2
        
            1. **Image - Topic Relevance:** You will be asked to pick a random cluster and assess the relevance of the randomly selected images in the cluster to the inferred topic.
            2. **Consistency of Images in a Topic:** You will be asked to pick a random cluster and rate how similar the images in the cluster are to each other.
            3. **Final Comments:** You can provide any additional comments or feedback at the end.
                 
            **IMPORTANT DISCLAIMER**
            
            The clusters and images you will see have been selected randomly and have not been reviewed or filtered for NSFW or sensitive content. If you come across such content and prefer not to work with it, please let us know.
        
            Please proceed by clicking "Next". You can always click "Previous" to go back and change your answers.
        """)

    elif pages.current == 2:
        st.header("1.1 Image - Topic Relevance")
        st.write("### - Given a random cluster, rate how relevant the screenshots in the cluster are to the topic.")
        cluster_key = list(st.session_state['random_clusters_1'].keys())[0]
        sampled_images = st.session_state['random_clusters_1'][cluster_key]            
        st.write(f"**Topic: {cluster_key}**")
        for i, row in sampled_images.iterrows():
            filename=row['Filename']
            st.write(f"**Screenshot {i+1}**")
            st.image(f"{imagedir}/{row['Filename']}", caption=f"Participant: {row['participant']}, App Name: {row['app']}, App Category: {row['app_category']}, Topic: {row['detailed_topic_1_labels']}", width=400)
            st.write(f"**Description:** {row['Document']}")
                
            st.write(f"**Topic: {cluster_key}**")

            survey.select_slider(
                    f"How relevant is **Screenshot {i+1}** to **the topic '{cluster_key}'**?",
                    options=["Highly Irrelevant",  "Irrelevant","Slightly Irrelevant", "No Answer", "Slightly Relevant", "Relevant", "Highly Relevant"],
                    value=st.session_state['responses'][f"relevance_{filename}"],
                    key=f"relevance_{filename}",
                    on_change= save_responses_to_temp_file
            )

            survey.select_slider(
                    f"How accurate is **the description** for **the Screenshot {i+1}?**",
                    options=["Highly Inaccurate", "Inaccurate", "Slightly Inaccurate", "No Answer", "Slightly Accurate", "Accurate", "Highly Accurate"],
                    value=st.session_state['responses'][f"description_{filename}"],
                    key=f"description_{filename}",
                    on_change=save_responses_to_temp_file
            )

            survey.text_area(
                    f"Please provide any additional comments for your rationale for your answer about Screenshot {i+1}:",
                    value=st.session_state['responses'][f"comment_{filename}"],
                    key=f"comment_{filename}",
                    on_change= save_responses_to_temp_file  # Use the session state key to store value
            )
            st.divider()

    
    elif pages.current == 3:
        st.header("1.2 Image - Topic Relevance")
        st.write("### - Given a random cluster, rate how relevant the screenshots in the cluster are to the topic.")
        cluster_key = list(st.session_state['random_clusters_1'].keys())[1]
        sampled_images = st.session_state['random_clusters_1'][cluster_key]            
        st.write(f"**Topic: {cluster_key}**")
        for i, row in sampled_images.iterrows():
            filename=row['Filename']
            st.write(f"**Screenshot {i+1}**")
            st.image(f"{imagedir}/{row['Filename']}", caption=f"Participant: {row['participant']}, App Name: {row['app']}, App Category: {row['app_category']}, Topic: {row['detailed_topic_1_labels']}", width=400)
            st.write(f"**Description:** {row['Document']}")
                
            st.write(f"**Topic: {cluster_key}**")

            survey.select_slider(
                    f"How relevant is **Screenshot {i+1}** to **the topic '{cluster_key}'**?",
                    options=["Highly Irrelevant",  "Irrelevant","Slightly Irrelevant", "No Answer", "Slightly Relevant", "Relevant", "Highly Relevant"],
                    value=st.session_state['responses'][f"relevance_{filename}"],
                    key=f"relevance_{filename}",
                    on_change= save_responses_to_temp_file
            )

            survey.select_slider(
                    f"How accurate is **the description** for **the Screenshot {i+1}?**",
                    options=["Highly Inaccurate", "Inaccurate", "Slightly Inaccurate", "No Answer", "Slightly Accurate", "Accurate", "Highly Accurate"],
                    value=st.session_state['responses'][f"description_{filename}"],
                    key=f"description_{filename}",
                    on_change=save_responses_to_temp_file
            )

            survey.text_area(
                    f"Please provide any additional comments for your rationale for your answer about Screenshot {i+1}:",
                    value=st.session_state['responses'][f"comment_{filename}"],
                    key=f"comment_{filename}",
                    on_change= save_responses_to_temp_file  # Use the session state key to store value
            )
            st.divider()  

    elif pages.current == 4:
        st.header("1.3 Image - Topic Relevance")
        st.write("### - Given a random cluster, rate how relevant the screenshots in the cluster are to the topic.")
        cluster_key = list(st.session_state['random_clusters_1'].keys())[2]
        sampled_images = st.session_state['random_clusters_1'][cluster_key]            
        st.write(f"**Topic: {cluster_key}**")
        for i, row in sampled_images.iterrows():
            filename=row['Filename']
            st.write(f"**Screenshot {i+1}**")
            st.image(f"{imagedir}/{row['Filename']}", caption=f"Participant: {row['participant']}, App Name: {row['app']}, App Category: {row['app_category']}, Topic: {row['detailed_topic_1_labels']}", width=400)
            st.write(f"**Description:** {row['Document']}")
                
            st.write(f"**Topic: {cluster_key}**")

            survey.select_slider(
                    f"How relevant is **Screenshot {i+1}** to **the topic '{cluster_key}'**?",
                    options=["Highly Irrelevant",  "Irrelevant","Slightly Irrelevant", "No Answer", "Slightly Relevant", "Relevant", "Highly Relevant"],
                    value=st.session_state['responses'][f"relevance_{filename}"],
                    key=f"relevance_{filename}",
                    on_change= save_responses_to_temp_file
            )

            survey.select_slider(
                    f"How accurate is **the description** for **the Screenshot {i+1}?**",
                    options=["Highly Inaccurate", "Inaccurate", "Slightly Inaccurate", "No Answer", "Slightly Accurate", "Accurate", "Highly Accurate"],
                    value=st.session_state['responses'][f"description_{filename}"],
                    key=f"description_{filename}",
                    on_change=save_responses_to_temp_file
            )

            survey.text_area(
                    f"Please provide any additional comments for your rationale for your answer about Screenshot {i+1}:",
                    value=st.session_state['responses'][f"comment_{filename}"],
                    key=f"comment_{filename}",
                    on_change= save_responses_to_temp_file  # Use the session state key to store value
            )
            st.divider()   

    elif pages.current == 5:
        st.header("1.4 Image - Topic Relevance")
        st.write("### - Given a random cluster, rate how relevant the screenshots in the cluster are to the topic.")
        cluster_key = list(st.session_state['random_clusters_1'].keys())[3]
        sampled_images = st.session_state['random_clusters_1'][cluster_key]            
        st.write(f"**Topic: {cluster_key}**")
        for i, row in sampled_images.iterrows():
            filename=row['Filename']
            st.write(f"**Screenshot {i+1}**")
            st.image(f"{imagedir}/{row['Filename']}", caption=f"Participant: {row['participant']}, App Name: {row['app']}, App Category: {row['app_category']}, Topic: {row['detailed_topic_1_labels']}", width=400)
            st.write(f"**Description:** {row['Document']}")
                
            st.write(f"**Topic: {cluster_key}**")

            survey.select_slider(
                    f"How relevant is **Screenshot {i+1}** to **the topic '{cluster_key}'**?",
                    options=["Highly Irrelevant",  "Irrelevant","Slightly Irrelevant", "No Answer", "Slightly Relevant", "Relevant", "Highly Relevant"],
                    value=st.session_state['responses'][f"relevance_{filename}"],
                    key=f"relevance_{filename}",
                    on_change= save_responses_to_temp_file
            )

            survey.select_slider(
                    f"How accurate is **the description** for **the Screenshot {i+1}?**",
                    options=["Highly Inaccurate", "Inaccurate", "Slightly Inaccurate", "No Answer", "Slightly Accurate", "Accurate", "Highly Accurate"],
                    value=st.session_state['responses'][f"description_{filename}"],
                    key=f"description_{filename}",
                    on_change=save_responses_to_temp_file
            )

            survey.text_area(
                    f"Please provide any additional comments for your rationale for your answer about Screenshot {i+1}:",
                    value=st.session_state['responses'][f"comment_{filename}"],
                    key=f"comment_{filename}",
                    on_change= save_responses_to_temp_file  # Use the session state key to store value
            )
            st.divider()   

    elif pages.current == 6:
        st.header("1.5 Image - Topic Relevance")
        st.write("### - Given a random cluster, rate how relevant the screenshots in the cluster are to the topic.")
        cluster_key = list(st.session_state['random_clusters_1'].keys())[4]
        sampled_images = st.session_state['random_clusters_1'][cluster_key]            
        st.write(f"**Topic: {cluster_key}**")
        for i, row in sampled_images.iterrows():
            filename=row['Filename']
            st.write(f"**Screenshot {i+1}**")
            st.image(f"{imagedir}/{row['Filename']}", caption=f"Participant: {row['participant']}, App Name: {row['app']}, App Category: {row['app_category']}, Topic: {row['detailed_topic_1_labels']}", width=400)
            st.write(f"**Description:** {row['Document']}")
                
            st.write(f"**Topic: {cluster_key}**")

            survey.select_slider(
                    f"How relevant is **Screenshot {i+1}** to **the topic '{cluster_key}'**?",
                    options=["Highly Irrelevant",  "Irrelevant","Slightly Irrelevant", "No Answer", "Slightly Relevant", "Relevant", "Highly Relevant"],
                    value=st.session_state['responses'][f"relevance_{filename}"],
                    key=f"relevance_{filename}",
                    on_change= save_responses_to_temp_file
            )

            survey.select_slider(
                    f"How accurate is **the description** for **the Screenshot {i+1}?**",
                    options=["Highly Inaccurate", "Inaccurate", "Slightly Inaccurate", "No Answer", "Slightly Accurate", "Accurate", "Highly Accurate"],
                    value=st.session_state['responses'][f"description_{filename}"],
                    key=f"description_{filename}",
                    on_change=save_responses_to_temp_file
            )

            survey.text_area(
                    f"Please provide any additional comments for your rationale for your answer about Screenshot {i+1}:",
                    value=st.session_state['responses'][f"comment_{filename}"],
                    key=f"comment_{filename}",
                    on_change= save_responses_to_temp_file  # Use the session state key to store value
            )
            st.divider()        
    
    elif pages.current == 7:
        st.header("2.1 Consistency of Images in a Topic")
        st.write("### - Given a random cluster, rate how similar the content of screenshots in the cluster are to each other.")
        cluster_key = list(st.session_state['random_clusters_2'].keys())[0]
        sampled_images = st.session_state['random_clusters_2'][cluster_key]  
        grid_images=st.session_state['grid_images'][cluster_key]   

        # Loop through the 5 sampled images
        for i, row in sampled_images.iterrows():
            filename=row['Filename']
            # Display 2 columns: Left for the single image, Right for the grid
            col1, col2 = st.columns([1, 1])

            # Left column: Single image for evaluation
            with col1:
                st.write(f"### Screenshot {i+1}")
                st.image(f"{imagedir}/{row['Filename']}", 
                            caption=f"Participant: {row['participant']}, App: {row['app']}, App Category: {row['app_category']}",
                            width=400)

            with col2:
                display_grid(grid_images)
            
            survey.select_slider(
                    f"How similar do you think **content of Screenshot {i+1} left** is to the **group of screenshots shown on the right?**",
                    options=["Highly Dissimilar", "Dissimilar", "Slightly Dissimilar", "No Answer", "Slightly Similar", "Similar", "Highly Similar"],
                    value=st.session_state['responses'][f"similarity_slider_{filename}"],
                    key=f"similarity_slider_{filename}",
                    on_change=  save_responses_to_temp_file
            )

            survey.text_area(
                    f"Please explain your reasoning. What factors influenced your similarity rating?",
                    value=st.session_state['responses'][f"similarity_comments_{filename}"],
                    key=f"similarity_comments_{filename}",
                    on_change= save_responses_to_temp_file
            )


            st.divider()


            

    elif pages.current == 8:
        st.header("2.1 Consistency of Images in a Topic")
        st.write("### - Given a random cluster, rate how similar the content of screenshots in the cluster are to each other.")
        cluster_key = list(st.session_state['random_clusters_2'].keys())[1]
        sampled_images = st.session_state['random_clusters_2'][cluster_key]  
        grid_images=st.session_state['grid_images'][cluster_key]          

        # Loop through the 5 sampled images
        for i, row in sampled_images.iterrows():
            filename=row['Filename']
            # Display 2 columns: Left for the single image, Right for the grid
            col1, col2 = st.columns([1, 1])

            # Left column: Single image for evaluation
            with col1:
                st.write(f"### Screenshot {i+1}")
                st.image(f"{imagedir}/{row['Filename']}", 
                            caption=f"Participant: {row['participant']}, App: {row['app']}, App Category: {row['app_category']}",
                            width=400)

            with col2:
                display_grid(grid_images)
            
            survey.select_slider(
                    f"How similar do you think **content of Screenshot {i+1} left** is to the **group of screenshots shown on the right?**",
                    options=["Highly Dissimilar", "Dissimilar", "Slightly Dissimilar", "No Answer", "Slightly Similar", "Similar", "Highly Similar"],
                    value=st.session_state['responses'][f"similarity_slider_{filename}"],
                    key=f"similarity_slider_{filename}",
                    on_change=  save_responses_to_temp_file
            )

            survey.text_area(
                    f"Please explain your reasoning. What factors influenced your similarity rating?",
                    value=st.session_state['responses'][f"similarity_comments_{filename}"],
                    key=f"similarity_comments_{filename}",
                    on_change= save_responses_to_temp_file
            )


            st.divider()

    elif pages.current == 9:
        st.header("2.1 Consistency of Images in a Topic")
        st.write("### - Given a random cluster, rate how similar the content of screenshots in the cluster are to each other.")
        cluster_key = list(st.session_state['random_clusters_2'].keys())[2]
        sampled_images = st.session_state['random_clusters_2'][cluster_key]
        grid_images=st.session_state['grid_images'][cluster_key]           
        # Loop through the 5 sampled images
        for i, row in sampled_images.iterrows():
            filename=row['Filename']
            # Display 2 columns: Left for the single image, Right for the grid
            col1, col2 = st.columns([1, 1])

            # Left column: Single image for evaluation
            with col1:
                st.write(f"### Screenshot {i+1}")
                st.image(f"{imagedir}/{row['Filename']}", 
                            caption=f"Participant: {row['participant']}, App: {row['app']}, App Category: {row['app_category']}",
                            width=400)

            with col2:
                display_grid(grid_images)
            
            survey.select_slider(
                    f"How similar do you think **content of Screenshot {i+1} left** is to the **group of screenshots shown on the right?**",
                    options=["Highly Dissimilar", "Dissimilar", "Slightly Dissimilar", "No Answer", "Slightly Similar", "Similar", "Highly Similar"],
                    value=st.session_state['responses'][f"similarity_slider_{filename}"],
                    key=f"similarity_slider_{filename}",
                    on_change=  save_responses_to_temp_file
            )

            survey.text_area(
                    f"Please explain your reasoning. What factors influenced your similarity rating?",
                    value=st.session_state['responses'][f"similarity_comments_{filename}"],
                    key=f"similarity_comments_{filename}",
                    on_change= save_responses_to_temp_file
            )


            st.divider()

    elif pages.current == 10:
        st.header("2.1 Consistency of Images in a Topic")
        st.write("### - Given a random cluster, rate how similar the content of screenshots in the cluster are to each other.")
        cluster_key = list(st.session_state['random_clusters_2'].keys())[3]
        sampled_images = st.session_state['random_clusters_2'][cluster_key]
        grid_images=st.session_state['grid_images'][cluster_key]           

        # Loop through the 5 sampled images
        for i, row in sampled_images.iterrows():
            filename=row['Filename']

            # Display 2 columns: Left for the single image, Right for the grid
            col1, col2 = st.columns([1, 1])

            # Left column: Single image for evaluation
            with col1:
                st.write(f"### Screenshot {i+1}")
                st.image(f"{imagedir}/{row['Filename']}", 
                            caption=f"Participant: {row['participant']}, App: {row['app']}, App Category: {row['app_category']}",
                            width=400)

            with col2:
                display_grid(grid_images)
            
            survey.select_slider(
                    f"How similar do you think **content of Screenshot {i+1} left** is to the **group of screenshots shown on the right?**",
                    options=["Highly Dissimilar", "Dissimilar", "Slightly Dissimilar", "No Answer", "Slightly Similar", "Similar", "Highly Similar"],
                    value=st.session_state['responses'][f"similarity_slider_{filename}"],
                    key=f"similarity_slider_{filename}",
                    on_change=  save_responses_to_temp_file
            )

            survey.text_area(
                    f"Please explain your reasoning. What factors influenced your similarity rating?",
                    value=st.session_state['responses'][f"similarity_comments_{filename}"],
                    key=f"similarity_comments_{filename}",
                    on_change= save_responses_to_temp_file
            )


            st.divider()

    elif pages.current == 11:
        st.header("2.1 Consistency of Images in a Topic")
        st.write("### - Given a random cluster, rate how similar the content of screenshots in the cluster are to each other.")
        cluster_key = list(st.session_state['random_clusters_2'].keys())[4]
        sampled_images = st.session_state['random_clusters_2'][cluster_key] 
        grid_images=st.session_state['grid_images'][cluster_key]        

        # Loop through the 5 sampled images
        for i, row in sampled_images.iterrows():
            filename=row['Filename']

            # Display 2 columns: Left for the single image, Right for the grid
            col1, col2 = st.columns([1, 1])

            # Left column: Single image for evaluation
            with col1:
                st.write(f"### Screenshot {i+1}")
                st.image(f"{imagedir}/{row['Filename']}", 
                            caption=f"Participant: {row['participant']}, App: {row['app']}, App Category: {row['app_category']}",
                            width=400)

            with col2:
                display_grid(grid_images)
            
            survey.select_slider(
                    f"How similar do you think **content of Screenshot {i+1} left** is to the **group of screenshots shown on the right?**",
                    options=["Highly Dissimilar", "Dissimilar", "Slightly Dissimilar", "No Answer", "Slightly Similar", "Similar", "Highly Similar"],
                    value=st.session_state['responses'][f"similarity_slider_{filename}"],
                    key=f"similarity_slider_{filename}",
                    on_change=  save_responses_to_temp_file
            )

            survey.text_area(
                    f"Please explain your reasoning. What factors influenced your similarity rating?",
                    value=st.session_state['responses'][f"similarity_comments_{filename}"],
                    key=f"similarity_comments_{filename}",
                    on_change= save_responses_to_temp_file
            )


            st.divider()

    elif pages.current == 12:
        #here, ask for final comments
        st.header("3. Final Comments")

        survey.text_area(
                "Any additional comments or feedback you would like to provide about the clustering and topic modeling results?",
                value=st.session_state['responses']["final_comments"],
                key="final_comments",
                on_change=save_responses_to_temp_file
            )

        st.write("#### Thank you for your participation! Please click 'Submit' to complete the survey.")
