import numpy as np
import torch

from .utils import clip_model, tokenizer
from search.models import ImageData, Tag
from search.v1.serializers import ImageDataSerializer
from search.v1.utils.utils import extract_image_metadata, generate_image_embedding, generate_image_text, generate_tags_from_caption, get_dominant_color, parse_tags


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("Invalid hex color format.")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def color_distance(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def compute_cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def direct_image_upload_search(image_file, user_tags=''):
    caption = generate_image_text(image_file)
    embedding_array = generate_image_embedding(image_file).astype(np.float32)
    metadata = extract_image_metadata(image_file)
    auto_tags = generate_tags_from_caption(caption)
    all_tags = set(auto_tags + parse_tags(user_tags))
    dominant_color = get_dominant_color(image_file)

    # Check if existing
    # existing_image = ImageData.objects.exclude(image_text=caption).exists()

    # if not existing_image:
    # Save to DB
    image_instance, _ = ImageData.objects.get_or_create(
        image=image_file,
        image_text=caption,
        image_embedding=embedding_array.tobytes(),
        image_meta=metadata,
        dominant_color=dominant_color
    )

    for tag in all_tags:
        tag_obj, _ = Tag.objects.get_or_create(name=tag)
        image_instance.image_tags.add(tag_obj)

    # Search similar images
    results = []
    for obj in ImageData.objects.exclude(id=image_instance.id):
        if not obj.image_embedding:
            continue
        db_emb = np.frombuffer(obj.image_embedding, dtype=np.float32)
        sim = compute_cosine_similarity(embedding_array, db_emb)
        results.append((sim, obj))

    results.sort(key=lambda x: x[0], reverse=True)
    top_related = [ImageDataSerializer(obj).data for _, obj in results[:5]]

    return {
        "uploaded_image": ImageDataSerializer(image_instance).data,
        "related_images": top_related
    }


def direct_text_search(query):
    prompt = f"a photo of {query}"
    with torch.no_grad():
        text_tokens = tokenizer([prompt])
        text_embedding = clip_model.encode_text(text_tokens).squeeze().numpy()
        text_embedding = text_embedding / np.linalg.norm(text_embedding)

    results = []
    for obj in ImageData.objects.all():
        if not obj.image_embedding:
            continue
        db_emb = np.frombuffer(obj.image_embedding, dtype=np.float32)
        db_emb = db_emb / np.linalg.norm(db_emb)
        sim = float(np.dot(text_embedding, db_emb))
        results.append((sim, obj))

    results.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            **ImageDataSerializer(obj).data,
            "similarity_score": round(sim, 4)
        }
        for sim, obj in results[:5]
    ]


def direct_color_search(color_query):
    try:
        query_rgb = hex_to_rgb(color_query)
    except ValueError:
        raise ValueError("Invalid hex color format. Use like '#aabbcc'")

    matches = []
    for obj in ImageData.objects.exclude(dominant_color__isnull=True):
        try:
            db_rgb = hex_to_rgb(obj.dominant_color)
            dist = color_distance(query_rgb, db_rgb)
            matches.append((dist, obj))
        except:
            continue

    matches.sort(key=lambda x: x[0])
    return [ImageDataSerializer(obj).data for _, obj in matches[:5]]
