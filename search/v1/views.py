from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from search.models import ImageData, Tag
from sentence_transformers import SentenceTransformer
import numpy as np
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
import numpy as np
from sentence_transformers import SentenceTransformer
import open_clip
import torch
from search.v1.serializers import ImageDataSerializer
from search.v1.utils import extract_image_metadata, generate_image_embedding, generate_image_text, generate_tags_from_caption, parse_tags


model = SentenceTransformer('clip-ViT-B-32')

tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model.eval()

class ImageDataViewset(viewsets.ModelViewSet):
    model = ImageData
    queryset = model.objects.all()
    serializer_class = ImageDataSerializer


@api_view(['GET'])
def search_images(request):
    query = request.GET.get('q')
    if not query:
        return Response({"error": "Query param 'q' is required"}, status=400)

    model = SentenceTransformer('clip-ViT-B-32')
    query_emb = model.encode(query)

    # Naive search - compare with stored embeddings
    results = []
    for image_obj in ImageData.objects.all():
        if not image_obj.image_embedding:
            continue
        db_emb = np.frombuffer(image_obj.image_embedding, dtype=np.float32)
        similarity = np.dot(query_emb, db_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(db_emb))
        results.append((similarity, image_obj))

    # Sort by similarity
    results.sort(key=lambda x: x[0], reverse=True)
    top_results = [ImageDataSerializer(obj).data for _, obj in results[:5]]

    return Response(top_results)


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_and_search(request):
    image_file = request.FILES.get('image')
    user_tags = request.data.get('tags', '')  # Optional extra tags

    if not image_file:
        return Response({"error": "Image file is required."}, status=400)

    # Generate AI features
    caption = generate_image_text(image_file)
    embedding_array = generate_image_embedding(image_file).astype(np.float32)
    embedding_bytes = embedding_array.tobytes()
    metadata = extract_image_metadata(image_file)
    auto_tags = generate_tags_from_caption(caption)
    all_tags = set(auto_tags + parse_tags(user_tags))

    # Save to DB
    image_instance = ImageData.objects.create(
        image=image_file,
        image_text=caption,
        image_embedding=embedding_bytes,
        image_meta=metadata
    )

    for tag_name in all_tags:
        tag_obj, _ = Tag.objects.get_or_create(name=tag_name)
        image_instance.image_tags.add(tag_obj)

    # Vector search (cosine similarity)
    query_emb = np.array(embedding_array, dtype=np.float32)
    results = []
    for obj in ImageData.objects.exclude(id=image_instance.id):
        if not obj.image_embedding:
            continue
        db_emb = np.frombuffer(obj.image_embedding, dtype=np.float32)
        sim = np.dot(query_emb, db_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(db_emb))
        results.append((sim, obj))

    results.sort(key=lambda x: x[0], reverse=True)
    # Selecting TOP 5 Relatable Images
    top_related = [ImageDataSerializer(obj).data for _, obj in results[:5]]

    return Response({
        "uploaded_image": ImageDataSerializer(image_instance).data,
        "related_images": top_related
    })

# @api_view(['POST'])
# def text_search(request):
#     query = request.data.get('query', '').strip()
#     if not query:
#         return Response({"error": "Field 'query' is required in the request body."}, status=400)

#     model = SentenceTransformer('clip-ViT-B-32')
#     query_emb = model.encode(query)

#     results = []
#     for image_obj in ImageData.objects.all():
#         if not image_obj.image_embedding:
#             continue
#         db_emb = np.frombuffer(image_obj.image_embedding, dtype=np.float32)
#         similarity = np.dot(query_emb, db_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(db_emb))
#         results.append((similarity, image_obj))

#     results.sort(key=lambda x: x[0], reverse=True)
#     top_results = [ImageDataSerializer(obj).data for _, obj in results[:5]]

#     return Response(top_results)
@api_view(['POST'])
def text_search(request):
    query = request.data.get('query', '').strip()
    if not query:
        return Response({"error": "Field 'query' is required."}, status=400)

    # Format prompt and encode text with OpenCLIP
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
    top_results = []
    for sim, obj in results[:5]:
        data = ImageDataSerializer(obj).data
        data['similarity_score'] = round(sim, 4)
        top_results.append(data)

    return Response({
        "query": query,
        "results": top_results
    })
