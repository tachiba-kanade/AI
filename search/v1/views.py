# from rest_framework import viewsets
# from rest_framework import permissions

# from search.models import ImageData
# from search.v1.serializers import ImageDataSerializer


# class ImageDataViewset(viewsets.ModelViewSet):
#     model = ImageData
#     queryset = model.objects.all()
#     serializer_class = ImageDataSerializer
#     # permission_classes = []


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

from search.v1.serializers import ImageDataSerializer
from search.v1.utils import extract_image_metadata, generate_image_embedding, generate_image_text, parse_tags


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
    tags_input = request.data.get('tags', '')

    print("Files: ", image_file)

    if not image_file:
        return Response({"error": "Image file is required."}, status=400)

    # Step 1: Generate all info
    embedding = generate_image_embedding(image_file)
    caption = generate_image_text(image_file)
    metadata = extract_image_metadata(image_file)

    # Step 2: Save the image
    image_instance = ImageData.objects.create(
        image=image_file,
        image_text=caption,
        image_embedding=embedding,
        image_meta=metadata
    )

    print("Views: ", image_instance)

    tag_names = parse_tags(tags_input)
    for tag in tag_names:
        tag_obj, _ = Tag.objects.get_or_create(name=tag)
        image_instance.image_tags.add(tag_obj)

    # Step 3: Search for similar images
    model = SentenceTransformer('clip-ViT-B-32')
    query_emb = np.frombuffer(embedding, dtype=np.float32)

    similar_images = []
    for obj in ImageData.objects.exclude(id=image_instance.id):
        if not obj.image_embedding:
            continue
        db_emb = np.frombuffer(obj.image_embedding, dtype=np.float32)
        sim = np.dot(query_emb, db_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(db_emb))
        similar_images.append((sim, obj))

    # Top 5 results
    similar_images.sort(key=lambda x: x[0], reverse=True)
    top_related = [ImageDataSerializer(img).data for _, img in similar_images[:5]]

    return Response({
        "uploaded_image": ImageDataSerializer(image_instance).data,
        "related_images": top_related
    })