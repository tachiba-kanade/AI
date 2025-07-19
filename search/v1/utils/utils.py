from PIL import Image as PILImage
import spacy
from PIL import Image
import open_clip
import torch
from PIL import Image
from colorthief import ColorThief
import io

from transformers import Blip2Processor, Blip2ForConditionalGeneration

tokenizer = open_clip.get_tokenizer('ViT-B-32')
# clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
clip_model.eval()

def generate_image_embedding(image_path_or_file):
    image = Image.open(image_path_or_file).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)  
    with torch.no_grad():
        embedding = clip_model.encode_image(image_tensor)
    return embedding.squeeze().numpy()

# Version 1.0
# from transformers import BlipProcessor, BlipForConditionalGeneration

# caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# def generate_image_text(image_path_or_file):
#     image = Image.open(image_path_or_file).convert("RGB")
#     inputs = caption_processor(images=image, return_tensors="pt")
#     out = caption_model.generate(**inputs)
#     caption = caption_processor.decode(out[0], skip_special_tokens=True)
#     print("Caption: --> ", caption)
#     return caption

# Load BLIP-2 with Flan-T5 XL (very detailed)

# Version 2.0
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=True)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_image_text(image_path_or_file):
    image = Image.open(image_path_or_file).convert("RGB")
    prompt = "Describe this image in detail with major colors in the photo."
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    output = model.generate(**inputs, max_new_tokens=250)
    description = processor.decode(output[0], skip_special_tokens=True)
    return description

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


nlp = spacy.load("en_core_web_sm")
def generate_tags_from_caption(caption):
    doc = nlp(caption)
    tags = set()
    for chunk in doc.noun_chunks:
        cleaned = chunk.text.strip().lower()
        if len(cleaned) > 1:
            tags.add(cleaned)

    print("Here is the tags from caption -----> ", list(tags))
    return list(tags)


def parse_tags(text):
    return [tag.strip().lower() for tag in text.split(',') if tag.strip()]

def get_dominant_color(image_file):
    try:
        if hasattr(image_file, 'read'):
            image_file.seek(0)
            color_thief = ColorThief(io.BytesIO(image_file.read()))
        else:
            color_thief = ColorThief(image_file)

        dominant_rgb = color_thief.get_color(quality=1)
        # Convert RGB to HEX
        hex_color = '#%02x%02x%02x' % dominant_rgb
        print("Dominant Color (HEX):", hex_color)
        return hex_color
    except Exception as e:
        print(f"Color extraction failed: {e}")
        return None