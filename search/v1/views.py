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
from search.v1.utils.search_utils import direct_color_search, direct_image_upload_search, direct_text_search
from search.v1.utils.utils import extract_image_metadata, generate_image_embedding, generate_image_text, generate_tags_from_caption, parse_tags, get_dominant_color
from django.http import QueryDict

from search.v1.utils.view_utils import extract_all_tags, find_similar_colors, find_similar_images, save_image_to_db

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
    user_tags = request.data.get('tags', '')

    if not image_file:
        return Response({"error": "Image file is required."}, status=400)

    result = direct_image_upload_search(image_file, user_tags)
    return Response(result)


@api_view(['POST'])
def text_search(request):
    query = request.data.get('query', '').strip()
    if not query:
        return Response({"error": "Field 'query' is required."}, status=400)

    result = direct_text_search(query)
    return Response({"query": query, "results": result})


@api_view(['GET'])
def color_search(request):
    color_query = request.GET.get('color', '').strip().lower()
    if not color_query:
        return Response({"error": "Provide a 'color' hex code or name"}, status=400)

    try:
        result = direct_color_search(color_query)
    except ValueError as e:
        return Response({"error": str(e)}, status=400)

    return Response({"query_color": color_query, "results": result})


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def input_detect(request):
    image = request.FILES.get('image')
    query = request.data.get('query')
    color = request.data.get('color')

    try:
        if image:
            img_results = direct_image_upload_search(image, [])
            return Response({"results": img_results})
        elif query:
            q_results = direct_text_search(query)
            return Response({"query": query, "results": q_results})
        elif color:
            color_result = direct_color_search(color)
            return Response({"query_color": color, "results": color_result})
    except Exception as e:
        return Response({"error": str(e)}, status=400)

    return Response({"error": "Provide at least one of: 'image', 'query', or 'color'"}, status=400)
