import requests
from PIL import Image, ImageFile
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import numpy as np
from torch.utils.data import DataLoader
import os
import csv
import warnings
import argparse
import time
import logging  # Add logging for diagnostics
from torchvision import transforms
from torch.profiler import profile, record_function, ProfilerActivity  # Profiling tools
import gc
import wandb
import sys


start_time=time.time()
# Set max split size to avoid large allocations and reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
warnings.filterwarnings("ignore")
log_dir = 'path/to/log/dir'
os.makedirs(log_dir, exist_ok=True)



# To handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True  # This tells PIL to load truncated images

# Argument Parsing
parser = argparse.ArgumentParser(description="LLaVA inference script")
parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing images.')
parser.add_argument('--output_file', type=str, required=True, help='Output CSV file to save results.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference.')
parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum number of new tokens to generate.')
parser.add_argument('--max_image_size', type=int, default=1024, help='Maximum size for the longest side of the input image.')
parser.add_argument('--prefetch_factor', type=int, default=4, help='how many batches should be in the buffer?')
parser.add_argument('--num_workers', type=int, default=8, help='how many workers of the cpu should load the data?')
args = parser.parse_args()

# Variables from arguments
image_folder = args.image_folder
batch_size = args.batch_size
max_new_tokens = args.max_new_tokens
max_image_size=args.max_image_size
prefetch_factor=args.prefetch_factor
num_workers=args.num_workers
output_file=args.output_file
pid=os.path.basename(image_folder)


device = "cuda"
timestamp=int(time.time())
logging.basicConfig(filename=f'{log_dir}/inference_{pid}_{timestamp}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
wandb.init(
    project="screenome_content_analysis",
    config={
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "max_image_size": max_image_size,
        "prefetch_factor":prefetch_factor,
        "num_workers":num_workers
    }
)

def resize_image(image, max_size):
    """Resize an image while maintaining the aspect ratio."""
    # Get the original dimensions
    original_width, original_height = image.size
    
    # Determine the scaling factor
    if max(original_width, original_height) > max_size:
        if original_width > original_height:
            new_width = max_size
            new_height = int((max_size / original_width) * original_height)
        else:
            new_height = max_size
            new_width = int((max_size / original_height) * original_width)
    else:
        # If the image is smaller than max_size, don't resize
        return image

    # Resize the image while maintaining aspect ratio
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

# Model loading
model_id = "llava-hf/llava-onevision-qwen2-7b-si-hf" #change this to the model id you want to use

torch.cuda.empty_cache()

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"

)
model=model.to("cuda")


processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "left"
model.eval()


# Start timing the entire script
script_start_time = time.time()
# Define your dataset class
class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, preloaded_images, texts):
        self.preloaded_images = preloaded_images
        self.texts = texts

    def __len__(self):
        return len(self.preloaded_images)

    def __getitem__(self, idx):
        #image_path = list(self.preloaded_images.keys())[idx]
        #image = self.preloaded_images[image_path]
        image_path, image = self.preloaded_images[idx]
        prompt = self.texts[idx]
        return {"id": idx, "image": image, "prompt": prompt, "path": image_path}

def collator(batch):
    ids = [b['id'] for b in batch]
    images = [b['image'] for b in batch]
    prompts = [b['prompt'] for b in batch]
    paths = [b['path'] for b in batch]  # Include path in collator output
    inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt")
    inputs['path'] = paths  # Add paths to inputs for tracking filenames in process_batch
    return ids, inputs

# Check if the output file exists and read processed filenames
processed_filenames = set()
if os.path.exists(output_file):
    with open(output_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        processed_filenames = {row[0] for row in reader}  # Collect already processed filenames
if not processed_filenames:
    with open(output_file, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Output"])

#Use sorted image paths
image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
image_paths = [path for path in image_paths if os.path.basename(path) not in processed_filenames]
if not image_paths:
    logging.info("All images have already been processed.")
    exit(0)

len_chunk=min(10000,len(image_paths))
image_paths = image_paths[:len_chunk]

# Prepare the prompt for the model, change this to the prompt you want to use
conversation_1 = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this screenshot in detail, using this output format: The screenshot displays [the content goes here, including verbatim text if present, or a specific description of what the text is about; if there is an image, describe exactly what it depicts]. Focus on explaining the exact content of any text or image. Include the app name if identifiable, but DO NOT include quantitative details, such as likes or status bar information.",


},
        ],
    }
]
prompt_1 = processor.apply_chat_template(conversation_1, add_generation_prompt=True)
prompts = np.repeat(prompt_1, len(image_paths)).tolist()
logging.info(f"Resizing and Preloading {len(image_paths)} images into memory...")
data_resize_preload_start_time = time.time()
#preloaded_images={}
preloaded_images = []
batch_load_times=[]
batch_inference_times=[]
batch_decode_times=[]

# Sequential image loading with resizing
for image_path in image_paths:
    try:
        image = Image.open(image_path).convert("RGB")
        #image = resize_image(image, max_size=max_image_size)  # Resize image to 1024px maximum, not needed since this is done earlier.
        #preloaded_images[image_path] = image
        preloaded_images.append((image_path, image))
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        continue

data_resize_preload_end_time = time.time()
data_resize_preload=data_resize_preload_end_time-data_resize_preload_start_time
logging.info(f"Finished preloading images {len(preloaded_images)}, and it took {data_resize_preload:.2f} seconds.")

datautils_prep_start_time = time.time()

dataset = ImageTextDataset(preloaded_images, prompts)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False,collate_fn=collator,prefetch_factor=prefetch_factor)

datautils_prep_end_time = time.time()
datautils_prep=datautils_prep_end_time-datautils_prep_start_time
logging.info(f"Data Utils Prep(dataset+data loader) took {datautils_prep:.2f} seconds.")

def process_batch(model, batch, processor, writer):
    """Process a single batch and write results using CUDA streams"""
    inference_start_time = time.time()
    
    model_inputs = {
            k: (v.to(device, non_blocking=True).to(torch.float16) if v.dtype == torch.float32 else v.to(device, non_blocking=True))
            for k, v in batch.items() if k not in ['ids','path']
        }
    
    # Run inference on compute_stream
    try: generate_ids = model.generate(**model_inputs, temperature=0, do_sample=False, max_new_tokens=max_new_tokens) #change hyperparameters here as well
    except RuntimeError as e:
        if "shape" in str(e):
            # Log problematic filenames due to shape error
            for path in batch['path']:
                #image_path = list(preloaded_images.keys())[id]
                #filename = os.path.basename(image_path)
                filename = os.path.basename(path)
                with open(f"path/to/error/log/file", "a") as log_file:
                    log_file.write(f"{filename} - Shape error: {e}\n")
            logging.error(f"Shape error encountered. Skipping batch due to shape mismatch.")
            return 0, 0, 0  # Return zero times to skip processing
    
    inference_end_time = time.time()
    inference_time_taken = inference_end_time - inference_start_time
    batch_inference_times.append(inference_time_taken)
    logging.info(f"Inference for batch took {inference_time_taken:.2f} seconds")

    batch_decode_start_time = time.time()
    text_outputs = processor.batch_decode(generate_ids.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    batch_decode_end_time = time.time()
    batch_decode_time = batch_decode_end_time - batch_decode_start_time
    batch_decode_times.append(batch_decode_time)
    logging.info(f"Decoding for batch took {batch_decode_time:.2f} seconds")

    write_output_start_time = time.time()
    for path, output in zip(batch['path'], text_outputs):
        cleaned_output = output.split('assistant', 1)[-1].replace('\n', ' ').strip()
        filename = os.path.basename(path)
        writer.writerow([filename, cleaned_output])
    write_output_end_time = time.time()
    write_output_time = write_output_end_time - write_output_start_time
    logging.info(f"Writing output for batch took {write_output_time:.2f} seconds")
    
    return inference_time_taken, batch_decode_time, write_output_time

with torch.inference_mode():
    with open(output_file, mode='a') as file:
        writer = csv.writer(file)
        for batch_idx, (ids, inputs) in enumerate(dataloader):
            batch_start_time = time.time()
            logging.info(f"Processing batch {batch_idx} with {len(ids)} images")
            # Directly process the batch
            inference_time, decode_time, write_time = process_batch(model, inputs, processor, writer)
            
            # Calculate data load time
            batch_end_time = time.time()
            batch_total_time = batch_end_time - batch_start_time
            data_load_time = batch_total_time - inference_time - decode_time - write_time
            batch_load_times.append(data_load_time)
            logging.info(f"Data load to GPU for batch {batch_idx} took {data_load_time:.2f} seconds")
            logging.info(f"Batch {batch_idx} total time: {batch_total_time:.2f} seconds")
            logging.info(f"Batch {batch_idx} processed ids: {ids}")

            # Log metrics to WandB
            wandb.log({
                "batch_idx": batch_idx,
                "batch_size": len(ids),
                "data_load_time": data_load_time,
                "inference_time": inference_time,
                "decode_time": decode_time,
                "write_time": write_time,
                "total_batch_time": batch_total_time,
            })
            if batch_idx % 20 == 0:  # Clear cache every 20 batches
                torch.cuda.empty_cache()


# Final summary logging
logging.info("Finished processing all images.")
logging.info(f"Average inference time per image: {np.mean(batch_inference_times):.2f} seconds")
logging.info(f"Average data loading time per image: {np.mean(batch_load_times):.2f} seconds")
logging.info(f"Average decoding time per image: {np.mean(batch_decode_times):.2f} seconds")

# Final WandB logging for average times
wandb.log({
    "final_avg_inference_time": np.mean(batch_inference_times),
    "final_avg_load_time": np.mean(batch_load_times),
    "final_avg_decode_time": np.mean(batch_decode_times),
})
wandb.finish()