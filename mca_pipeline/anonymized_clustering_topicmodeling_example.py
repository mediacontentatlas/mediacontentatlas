from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import transformers
from huggingface_hub import login
from datasets import Dataset
from bertopic import BERTopic
from torch import bfloat16
import numpy as np
import pandas as pd


login(token="your_token")


def load_embeddings(file_path):
    npzfile = np.load(file_path, allow_pickle=True)
    filenames = npzfile['filenames'] if 'filenames' in npzfile else None
    embeddings = npzfile['embeddings']
    return filenames, embeddings

file_path = "/path/to/descriptionembeddings"  # Replace with your path
filenames, embeddings = load_embeddings(file_path)

data=pd.read_csv("/path/to/csv/with/filenamesanddescriptions",low_memory=False)

data_filenames=data['Filename']
docs=data['Output']
#Based on your choice of model, you can use the following code to load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    trust_remote_code=True,
    device_map='auto',
)
model.eval()
model.resize_token_embeddings(len(tokenizer))

embedding_model=SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

generator = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    temperature=0.1,
    max_new_tokens=500,
    repetition_penalty=1.1,
    batch_size=4
)

system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling topics.
<</SYS>>
"""
example_prompt = """
I have a topic that contains the following documents:
- The screenshot displays a website where it says traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- The screenshot features an image of meat and the captions read: Meat, but especially beef, is the word food in terms of emissions.
- The screenshot shows a quote on Twitter: "Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one".

The topic is described by the following keywords: 'meat, beef, eating, emissions, food, health, image, processed, climate change, Twitter quote'.

Based on the information about the topic above, please create a short and specific label of this topic, in English. Make sure you to only return the label and nothing more.
[/INST] Environmental impacts of eating meat
"""

main_prompt = """
[INST]
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short and specific label of this topic, in English. Make sure you to only return the label and nothing more.
[/INST]
"""
prompt = system_prompt + example_prompt + main_prompt


#change your parameters for UMAP and HDBSCAN based on your data(one way to find improve on the parameters can be using HDBSCAN's validity index)
umap_model= UMAP(n_components=5,n_neighbors=15, n_epochs=1000,min_dist=0.00,metric='cosine',random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
reduced_embeddings=UMAP(n_components=20,n_neighbors=15,min_dist=0.00,n_epochs=1000,metric='cosine',random_state=42).fit_transform(embeddings)

# Representation Models
keybert = KeyBERTInspired(nr_repr_docs=20)
llama2 = TextGeneration(generator, prompt=prompt,nr_docs=10, diversity=0.1)

representation_model = {
    "KeyBERT": keybert,
    "Llama2": llama2
}

# BERTopic Model
topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=CountVectorizer(stop_words="english"),
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    representation_model=representation_model,
    top_n_words=10,
    verbose=True
)
topics, probs = topic_model.fit_transform(docs,reduced_embeddings)
topic_model.get_topic_info().to_csv("path/to/save/topicinfo.csv",index=False)