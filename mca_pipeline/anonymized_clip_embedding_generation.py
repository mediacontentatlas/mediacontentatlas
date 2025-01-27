import clip
import os
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import pathlib
from torch import nn
import numpy as np
from torch.utils.data import ChainDataset
import pickle
import argparse
import timm


def get_model(model_name, device='cuda'):
    if "clip" in model_name:
        models = ['RN50x16', 'ViT-B/16', 'ViT-L/14']
        model, _ = clip.load(models[2], device=device)
        encode_image = model.encode_image
    else:
        model = timm.create_model(model_name, pretrained=True)
        encode_image = lambda x: model.forward_features(x).mean(-1).mean(-1)
    return model.to(device), encode_image


# https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/9
class SquarePad:
    def __call__(self, image):
        image_shape = image[0].shape
        max_wh = max(image_shape)
        p_left, p_top = [(max_wh - s) // 2 for s in image_shape]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image_shape, [p_left, p_top])]
        padding = (p_top, p_left, p_bottom, p_right)
        return F.pad(image, padding, 0, 'constant')

transform = transforms.Compose([
    transforms.Resize(size=224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    SquarePad(),
    transforms.Resize(size=224),
])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('model', help='the name of the model to use')
    parser.add_argument('batch_size', default=256, type=int, help='batchsize')
    args = parser.parse_args()
    device = 'cuda'

    model_name = args.model
    model, encode_image = get_model(model_name, device=device)
    roots = ['path1', 'path2', 'path3'] # Add the paths to the folders containing the images here
    for root in roots:
        all_image_dataset = ImageFolder(root='/home/' + root, transform=transform)
        img_order = all_image_dataset.imgs

        root_dir = f"saved_data/{model_name}/{root}"

        os.makedirs(root_dir, exist_ok=True)

        cur_files = os.listdir(root_dir)
        max_idx = max([int(cur_file.split(".")[0].split("_")[-1]) for cur_file in cur_files] + [0])

        if max_idx != 0:
            all_image_dataset = torch.utils.data.Subset(all_image_dataset, range(max_idx, len(all_image_dataset)))
            print(f"Starting from {max_idx} of {len(all_image_dataset)}")

        all_image_dataloader = DataLoader(all_image_dataset, batch_size=args.batch_size, num_workers=12)

        with open(f'saved_data/{model_name}/{root}_filenames.pkl', 'wb') as fp:
            pickle.dump(img_order, fp)

        def save_batch_array(accumulated_data):
            return np.savez_compressed(
                f"{root_dir}/{file_counter * save_batch}_{file_counter * save_batch + len(accumulated_data)}.npz",
                embeddings=accumulated_data
            )

        save_batch = 65536
        file_counter = max_idx // save_batch
        accumulated_data = None
        with torch.no_grad():
            for image_idx, (image, _) in enumerate(tqdm(all_image_dataloader)):
                image = image.to(device)
                
                # image = nn.functional.interpolate(image, 224)
                image_encodings = encode_image(image).cpu()
                if accumulated_data is None:
                    accumulated_data = [image_encodings.clone()]
                else:
                    accumulated_data.append(image_encodings)
                
                if len(accumulated_data) >= save_batch // args.batch_size:
                    accumulated_data = torch.cat(accumulated_data).contiguous()
                    save_batch_array(accumulated_data)
                    accumulated_data = None
                    file_counter += 1
            if accumulated_data is not None:
                accumulated_data = torch.cat(accumulated_data).contiguous()
                save_batch_array(accumulated_data)
                accumulated_data = None
                file_counter += 1