import numpy as np
from requests import Response
import torch

from search.models import ImageData
from search.v1.serializers import ImageDataSerializer
from search.v1.utils.utils import generate_image_text, generate_tags_from_caption, parse_tags

# ---- Tagging & Metadata ----
def extract_all_tags(image_file, user_tags):
    caption = generate_image_text(image_file)
    auto_tags = generate_tags_from_caption(caption)
    all_tags = set(auto_tags + parse_tags(user_tags))
    return caption, all_tags

def save_image_to_db(image_file, caption, embedding_array, metadata, dominant_color, all_tags):
    image_instance = ImageData.objects.create(
        image=image_file,
        image_text=caption,
        image_embedding=embedding_array.tobytes(),
        image_meta=metadata,
        dominant_color=dominant_color
    )

    for tag in all_tags:
        tag_obj, _ = torch.Tag.objects.get_or_create(name=tag)
        image_instance.image_tags.add(tag_obj)

    return image_instance

# ---- Embeddings & Similarity ----
def compute_cosine_similarity(vec1, vec2):
    if not vec1.any() or not vec2.any():
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def find_similar_images(query_emb, exclude_id=None, top_k=5):
    query_emb = np.array(query_emb, dtype=np.float32)
    results = []

    images = ImageData.objects.exclude(id=exclude_id) if exclude_id else ImageData.objects.all()
    for obj in images:
        if not obj.image_embedding:
            continue
        db_emb = np.frombuffer(obj.image_embedding, dtype=np.float32)
        sim = compute_cosine_similarity(query_emb, db_emb)
        results.append((sim, obj))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

# ---- Color Conversion ----
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("Invalid hex format.")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def color_distance(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

def find_similar_colors(query_color, top_k=5):
    try:
        query_rgb = hex_to_rgb(query_color)
    except:
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
    return matches[:top_k]

def direct_color_search(color_query):
    try:
        matches = find_similar_colors(color_query.strip().lower())
    except ValueError as e:
        return Response({"error": str(e)}, status=400)

    results = [ImageDataSerializer(obj).data for _, obj in matches]

    return Response({
        "query_color": color_query,
        "results": results
    })
