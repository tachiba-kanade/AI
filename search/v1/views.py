from rest_framework import viewsets
from rest_framework import permissions

from search.models import ImageData
from search.v1.serializers import ImageDataSerializer


class ImageDataViewset(viewsets.ModelViewSet):
    model = ImageData
    queryset = model.objects.all()
    serializer_class = ImageDataSerializer
    # permission_classes = []
