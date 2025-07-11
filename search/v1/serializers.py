from rest_framework import serializers
from ..models import ImageData


class ImageDataSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ImageData
        fields = ['image_text', 'image',
            'image_embedding',
            'image_tags',
            'created_at',
            'updated_at',
            ]
