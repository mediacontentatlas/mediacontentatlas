from huggingface_hub import login
import pandas as pd
import time  # For timing
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pickle
from sklearn.preprocessing import MinMaxScaler
from itertools import islice

login(token="your_token_here")

       
def load_embeddings_from_npz(filename, to_cuda=True):
    data = np.load(filename, allow_pickle=True)
    filenames = data['filenames']
    embeddings = data['embeddings']
    if to_cuda:
        embeddings = torch.tensor(embeddings, device='cuda')
    embeddings_dict = {fname: embedding for fname, embedding in zip(filenames, embeddings)}
    return embeddings_dict

def load_embeddings(filename, to_cuda=True):
    """
    Load embeddings from a pickle file containing sequentially dumped tuples.

    Args:
        filename (str): Path to the pickle file containing tuples of (filename, embedding).
        to_cuda (bool): Whether to move embeddings to CUDA.

    Returns:
        dict: A dictionary mapping filenames to their embeddings.
    """
    print(f"Loading embeddings from {filename}...")
    embeddings_dict = {}
    try:
        with open(filename, 'rb') as file:
            while True:
                try:
                    filename, embedding = pickle.load(file)
                    if to_cuda:
                        embedding = torch.tensor(embedding, device='cuda')
                    embeddings_dict[filename] = embedding
                except EOFError:
                    break
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except pickle.UnpicklingError:
        print("Error: Failed to unpickle the file. Ensure it's a valid pickle file.")
    except Exception as e:
        print(f"An unexpected error occurred while loading embeddings: {e}")
    print(f"Loaded {len(embeddings_dict)} embeddings.")
    return embeddings_dict


def calculate_llava_similarities(model, query, embeddings_dict, top_k):
    """
    Calculate similarities between a query embedding and precomputed embeddings in a file.
    """
    print(f"Calculating LLAVA similarities for query: '{query}'")
    start_time = time.time()

    query_embedding = torch.tensor(model.encode(query), device='cuda')  # Move query to CUDA

    similarities = {
        filename: torch.nn.functional.cosine_similarity(query_embedding, embedding, dim=0).item()
        for filename, embedding in embeddings_dict.items()
    }

    # Normalize similarities
    similarity_values = np.array(list(similarities.values())).reshape(-1, 1)
    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(similarity_values).flatten()
    normalized_similarities = {
        filename: normalized_value for filename, normalized_value in zip(similarities.keys(), normalized_values)
    }

    # Sort and select top_k
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_k_similarities = dict(islice(sorted_similarities, top_k))
    top_k_normalized_similarities = {
        filename: normalized_similarities[filename] for filename in top_k_similarities.keys()
    }
    top_k_filenames = list(top_k_similarities.keys())

    print(f"Finished LLAVA similarity calculation for query: '{query}' in {time.time() - start_time:.2f} seconds.")
    return top_k_filenames, top_k_normalized_similarities, top_k_similarities


def calculate_clip_similarities(clip_model, embeddings_dict, query, top_k):
    """
    Calculate similarities between a query embedding and CLIP embeddings from a file.
    """
    print(f"Calculating CLIP similarities for query: '{query}'")
    start_time = time.time()

    text_embedding = torch.tensor(clip_model.encode(query), device='cuda')

    similarities = {
        filename: torch.nn.functional.cosine_similarity(text_embedding, embedding, dim=0).item()
        for filename, embedding in embeddings_dict.items()
    }

    # Normalize similarities
    similarity_values = np.array(list(similarities.values())).reshape(-1, 1)
    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(similarity_values).flatten()
    normalized_similarities = {
        filename: normalized_value for filename, normalized_value in zip(similarities.keys(), normalized_values)
    }

    # Sort and select top_k
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_k_similarities = dict(islice(sorted_similarities, top_k))
    top_k_normalized_similarities = {
        filename: normalized_similarities[filename] for filename in top_k_similarities.keys()
    }
    top_k_filenames = list(top_k_similarities.keys())

    print(f"Finished CLIP similarity calculation for query: '{query}' in {time.time() - start_time:.2f} seconds.")
    return top_k_filenames, top_k_normalized_similarities, top_k_similarities


def evaluate_models(llava_model, clip_model, queries, llava_embeddings, clip_embeddings, top_k=20):
    """
    Evaluate both LLAVA and CLIP models for the given queries and save results.
    """
    print(f"Evaluating models for {len(queries)} queries...")
    results = {'llava': {}, 'clip': {}}

    for i, query in enumerate(queries):
        print(f"Processing query {i + 1}/{len(queries)}: '{query}'")

        llava_top_k_filenames, llava_top_k_norm_similarities, llava_top_k_similarities = calculate_llava_similarities(
            llava_model, query, llava_embeddings, top_k
        )

        clip_top_k_filenames, clip_top_k_norm_similarities, clip_top_k_similarities = calculate_clip_similarities(
            clip_model, clip_embeddings, query, top_k
        )

        results['llava'][query] = {
            filename: {
                "similarity": llava_top_k_similarities[filename],
                "normalized_similarity": llava_top_k_norm_similarities[filename],
            }
            for filename in llava_top_k_similarities
        }

        results['clip'][query] = {
            filename: {
                "similarity": clip_top_k_similarities[filename],
                "normalized_similarity": clip_top_k_norm_similarities[filename],
            }
            for filename in clip_top_k_similarities
        }

    print("Model evaluation completed.")
    return results



def convert_results_to_serializable(results_dict):
    """
    Recursively convert a results dictionary into a JSON-serializable format.

    Args:
        results_dict (dict): Dictionary of results to convert.

    Returns:
        dict: JSON-serializable dictionary.
    """
    def convert_value(value):
        if isinstance(value, torch.Tensor):
            return value.item() if value.dim() == 0 else value.tolist()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (dict, list, tuple)):
            return convert_results_to_serializable(value)  # Recursive call for nested structures
        elif isinstance(value, (float, int, str, bool)) or value is None:
            return value  # Already serializable
        else:
            return str(value)  # Fallback for other types

    if isinstance(results_dict, dict):
        return {key: convert_value(val) for key, val in results_dict.items()}
    elif isinstance(results_dict, (list, tuple)):
        return [convert_value(item) for item in results_dict]
    else:
        return convert_value(results_dict)

def save_embeddings_as_npz(embeddings_dict, filename):
    filenames = list(embeddings_dict.keys())
    embeddings = np.array([
        embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding
        for embedding in embeddings_dict.values()
    ])
    np.savez_compressed(filename, filenames=filenames, embeddings=embeddings)

def save_results_to_json(results_dict, filename):
    """
    Save results dictionary to a JSON file after converting it into a serializable format.
    """
    print(f"Saving results to {filename}...")
    try:
        serializable_results = convert_results_to_serializable(results_dict)
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"Results successfully saved to {filename}.")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

def live_query(llava_model, clip_model, query, llava_embeddings, clip_embeddings, top_k=10):
    """
    Handle live queries for both LLAVA and CLIP models and return top_k results.

    Args:
        llava_model: The LLAVA model used for similarity calculation.
        clip_model: The CLIP model used for similarity calculation.
        query (str): The user's query string.
        llava_embeddings_file (str): Path to the LLAVA embeddings file.
        clip_embeddings_file (str): Path to the CLIP embeddings file.
        top_k (int): Number of top results to return.

    Returns:
        dict: Results for the query, including raw and normalized similarities.
    """
    print(f"Processing live query: '{query}' with top_k={top_k}")

    # Calculate similarities using LLAVA
    llava_top_k_filenames, llava_top_k_norm_similarities, llava_top_k_similarities = calculate_llava_similarities(
        llava_model, query, llava_embeddings, top_k
    )

    # Calculate similarities using CLIP
    clip_top_k_filenames, clip_top_k_norm_similarities, clip_top_k_similarities = calculate_clip_similarities(
        clip_model, clip_embeddings, query, top_k
    )

    # Format results for the query
    results = {
        'llava': {
            filename: {
                "similarity": llava_top_k_similarities[filename],
                "normalized_similarity": llava_top_k_norm_similarities[filename],
            }
            for filename in llava_top_k_similarities
        },
        'clip': {
            filename: {
                "similarity": clip_top_k_similarities[filename],
                "normalized_similarity": clip_top_k_norm_similarities[filename],
            }
            for filename in clip_top_k_similarities
        }
    }

    print(f"Finished processing live query: '{query}'")
    return results


llava_embeddings_file="/path/to/llava_embeddings.npz"
clip_embeddings_file="/path/to/clip_embeddings.npz"
llava_model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True, device='cuda')#change to your model
llava_embeddings = load_embeddings_from_npz(llava_embeddings_file, to_cuda=True)  # Load embeddings to CUDA
clip_model = SentenceTransformer("clip-ViT-L-14", trust_remote_code=True, device='cuda') #change to your model
clip_embeddings = load_embeddings_from_npz(clip_embeddings_file, to_cuda=True)  # Load embeddings to CUDA

queries=["food","blood","guns","cat","nature","alcohol","Trump","Biden","gambling","Coca Cola","depression","violent","political","health","bullying","shopping","therapy","substance abuse","finance","influencer"] #change to your queries
queries = ["An image with "+i+" content" for i in queries] # Add prefix to queries
top_k=20
# Evaluate models
start_time = time.time()
results = evaluate_models(llava_model, clip_model, queries, llava_embeddings, clip_embeddings,top_k=top_k)
print(f"Total evaluation time: {time.time() - start_time:.2f} seconds.")

# Save results to JSON
save_results_to_json(results, "retrieval_evaluation_results.json") #change to your path
print(json.dumps(results, indent=4))