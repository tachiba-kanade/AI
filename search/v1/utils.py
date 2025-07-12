from PIL import Image as PILImage
from sentence_transformers import SentenceTransformer
import torch
from torchvision import transforms
import numpy as np
import PIL
import io
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

clip_model = SentenceTransformer('clip-ViT-B-32')


def generate_image_embedding(image_path_or_file):
    image = PILImage.open(image_path_or_file).convert("RGB")
    embedding = clip_model.encode(image)
    print("Embedds: ", embedding)
    return embedding.tobytes()  # To store in BinaryField


# Load once at module level (not inside function for speed)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# # More Intensive Generation - requires more memory
# processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto", torch_dtype=torch.float16)

def generate_image_text(image_path_or_file):
    image = Image.open(image_path_or_file).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print("Caption: --> ", caption)
    return caption


def extract_image_metadata(image_file):
    try:
        image = PILImage.open(image_file)
        print({
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
        })
        return {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
        }

    except Exception as e:
        return {"error": str(e)}


def parse_tags(text):
    # Naive tagging: split words, remove stopwords etc.
    # For now just split commas
    return [tag.strip().lower() for tag in text.split(',') if tag.strip()]