from huggingface_hub import login
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import os
import argparse
import torch
# Authenticate Hugging Face Hub
login(token="your_token_here")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Function to remove specific keywords from descriptions
def remove_keywords(file):
    remove_keywords = ['smartphone', 'screen', 'cell', 'cellphone', 'display', 'contain', 'phone', 'Screenshot', 
                       'displays', 'contains', 'screenshots', 'screenshot']
    file['Output'] = file['Output'].str.replace('|'.join(remove_keywords), '', regex=True)
    return file['Output']


# Function to precompute embeddings and store them
def precompute_and_store_embeddings(model, input_filename, output_base_directory,batch_size):
    df = pd.read_csv(input_filename, low_memory=False)
    remove_keywords(df)
    
    # Set the output filename
    output_filename = os.path.basename(input_filename).replace(".csv", "_embeddings_gte-large.pkl")

    
    # Ensure the output directory exists
    os.makedirs(output_base_directory, exist_ok=True)
    output_filepath = os.path.join(output_base_directory, output_filename)
    if os.path.exists(output_filepath):
        print(f"Embeddings already exist for {input_filename}. Skipping...")
        return

    # Save embeddings
    with open(output_filepath, 'wb') as f:
        # Use batch encoding for efficiency
        # Remove keywords if needed
        df['Output'] = remove_keywords(df)
        descriptions = df['Output'].tolist()
        filenames = df['Filename'].tolist()
        embeddings = model.encode(descriptions, batch_size=batch_size, show_progress_bar=True, device='cuda')
        
        for filename, embed in zip(filenames, embeddings):
            pickle.dump((filename, embed), f)
    print(f"Embeddings for {input_filename} precomputed and stored in {os.path.join(output_base_directory, output_filename)}")


# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compute and store embeddings for CSV files.") #the descriptions were in the 'Output' column of the CSV files for each participant
    parser.add_argument("--input_dir", required=True, help="Path to the input directory containing CSV files.")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory to store embeddings.")
    parser.add_argument("--batch_size",type=int,required=True,help="What should be the batch size?")
    args = parser.parse_args()

    # Collect CSV files from the input directory
    csv_files = [os.path.join(args.input_dir, file) for file in os.listdir(args.input_dir) if file.endswith(".csv")]

    # Load and process with Model 1
    print("Loading Model 1: Alibaba-NLP/gte-large-en-v1.5")
    embedding_model_1 = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True, device='cuda')
    for csv_file in csv_files:
        precompute_and_store_embeddings(embedding_model_1, csv_file, args.output_dir,args.batch_size)
        torch.cuda.empty_cache()
    del embedding_model_1  # Clear GPU memory
    torch.cuda.empty_cache()
    print("Model 1 processing completed. GPU memory cleared.")

if __name__ == "__main__":
    main()
