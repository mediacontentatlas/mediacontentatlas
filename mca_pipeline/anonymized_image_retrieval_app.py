from huggingface_hub import login
import streamlit as st
import json
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from itertools import islice
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from PIL import Image
import os
import csv
from datetime import datetime
login(token="your_hf_token") #change this token
image_dir="/path/to/images" #change this path
torch.cuda.empty_cache()
feedback_file = "freeformretrieval-feedback.csv" #change this path

# Load embeddings
def load_embeddings_from_npz(filename, to_cuda=True):
    data = np.load(filename, allow_pickle=True)
    filenames = data['filenames']
    embeddings = data['embeddings']
    if to_cuda:
        embeddings = torch.tensor(embeddings, device='cuda')
    embeddings_dict = {fname: embedding for fname, embedding in zip(filenames, embeddings)}
    return embeddings_dict
def save_feedback(query, feedback):
    # Check if the file exists and write header if not
    file_exists = os.path.exists(feedback_file)
    with open(feedback_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "query", "feedback"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), query, feedback])

# Similarity calculation
def calculate_similarities(model, query, embeddings_dict, top_k):
    query_embedding = torch.tensor(model.encode(query), device='cuda')
    similarities = {
        filename: torch.nn.functional.cosine_similarity(query_embedding, embedding, dim=0).item()
        for filename, embedding in embeddings_dict.items()
    }
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return dict(islice(sorted_similarities, top_k))


# Initialize the app
st.set_page_config(page_title="MCA Free-Form Screenshot Retrieval", page_icon="🔎", layout="wide", initial_sidebar_state="auto", menu_items=None)
# Persist results in session state
if "results" not in st.session_state:
    st.session_state["results"] = {}

st.title("Free-Form Screenshot Retrieval using LLaVA and CLIP embeddings")
st.write("This app allows you to perform real-time retrieval using your query and pre-computed embeddings.")
with st.expander("Section 1: READ ME"):
    st.subheader("Important Notes")
    st.write("""
    1. The output of this app will be the top-k most similar screenshots to your query in the 1.12M screenshots embedded. 
    - **Please ensure no one is looking at your screen, and do not take screenshots of the results.**
    2. You might encounter **NSFW content**, even though you are not searching for it, so be aware of this possibility.
    3. The results have not been rigorously tested for accuracy and should be used at your own risk.
    """)
    st.subheader("What to expect")
    st.write("""
    1. **Query**: Enter a query that best describes the content of the screenshot you are looking for. (e.g, an image of a healthy meal, messaging about elections, uber ride pick up search)
    2. **Number of top results (k)**: Choose the number of top results you want to see.
    3. **Querying takes time**: In the background we have limited memory, and searching takes a bit time, so after you query, give it 5 mins, then you can download & view your results.
    4. **Download your results**: Make sure download the json file of the results for future use.
    5. **View your results**: If you want to investigate the results you downloaded, upload the json file, and don't forget to leave feedback if you see any interesting results.
    """)
    read_acknowledged = st.checkbox("I have read and understood the above information.")
    if not read_acknowledged:
        st.warning("You must acknowledge that you've read and understood the information above before proceeding.")
if read_acknowledged:
        with st.expander("Section 2: Query"):

            # User inputs
            query = st.text_input("Enter your query:", "")
            top_k = st.slider("Number of top results (k):", min_value=1, max_value=100, value=10)
            # Results container
            results_container = st.container()

            # Run query button
            if st.button("Run Query"):
                if not query.strip():
                    st.error("Please enter a valid query!")
                else:
                    results = {}
                    with st.spinner(f"Loading model 1 and 1.2 million embeddings..."):
                        model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True, device='cuda') #change this model to model of your choice
                        llava_embeddings = load_embeddings_from_npz("/path/to/precomputed/llava_embeddings.npz") #change this path
                    with st.spinner(f"Similarities are being calculated, picking top results from 1.2 Million images... Be patient :) "):
                        st.session_state["results"]["llava"] = calculate_similarities(model, query, llava_embeddings, top_k)
                        st.success("Llava's top results are found!")
                        torch.cuda.empty_cache()
                    with st.spinner(f"Loading model 2 and 1.2 million embeddings..."):
                        model = SentenceTransformer("clip-ViT-L-14", trust_remote_code=True, device='cuda') #change this model to model of your choice
                        clip_embeddings = load_embeddings_from_npz("/path/to/precomputed/clip_embeddings.npz") #change this path
                    with st.spinner(f"Similarities are being calculated, picking top results from 1.2 Million images... Be patient :) "):
                        st.session_state["results"]["clip"] = calculate_similarities(model, query, clip_embeddings, top_k)
                        st.success("CLIP's top results are found!")
                        st.success("Results are ready!")
                        torch.cuda.empty_cache()

                        st.download_button(
                            label="Download Results as JSON",
                            data=json.dumps(st.session_state["results"], indent=4),
                            file_name=f"retrieval_results_{query.replace(' ', '_')}.json",
                            mime="application/json"
                        )
        with st.expander("Section 3: View the Results"):
            if st.session_state["results"]:
                st.write("The images retrieved based on your query are shown below:")
                for model_name, model_results in st.session_state["results"].items():
                    st.write(f"### Results from {model_name.upper()} model")
                    for filename, score in model_results.items():
                        try:
                            image = Image.open(os.path.join(image_dir,filename))
                            st.image(image, caption=f"{filename} (Score: {score:.2f})",width=300)
                        except FileNotFoundError:
                            st.warning(f"Image {filename} not found.")
            else:
                st.write("No results yet, make sure to complete the query first.")

        with st.expander("Section 4: Upload Results to see Images"):
            uploaded_file = st.file_uploader("Upload your results JSON file", type=["json"])
            
            if uploaded_file is not None:
                try:
                    # Load JSON file
                    uploaded_results = json.load(uploaded_file)
                    
                    st.write("The images from your uploaded results are shown below:")
                    for model_name, model_results in uploaded_results.items():
                        st.write(f"### Results from {model_name.upper()} model")
                        for filename, score in model_results.items():
                            # Load and display image
                            try:
                                image = Image.open(os.path.join(image_dir,filename))
                                st.image(image, caption=f"{filename} (Score: {score:.2f})",width=300)
                            except FileNotFoundError:
                                st.warning(f"Image {filename} not found.")
                except json.JSONDecodeError:
                    st.error("Invalid JSON file. Please upload a valid results JSON.")
            else: st.warning("No json file is uploaded yet or the uploaded file is empty.")

        with st.expander("Section 5: Feedback"):
            st.write("We'd love to hear your thoughts on the output and your experience with this app. If you have specific feedback or issues, please include the filename and your query information so we can investigate later.")
            feedback = st.text_area("Any comments or suggestions?", "")
            if st.button("Submit Feedback"):
                if feedback:
                    # Save feedback to CSV
                    save_feedback(query, feedback)
                    st.success("Thank you for your feedback!")
                else:
                    st.warning("Please provide your feedback before submitting.")