from rest_framework import serializers
from search.models import ImageData, Tag
from search.v1.utils import extract_image_metadata, generate_image_embedding, generate_image_text, parse_tags


class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = ['id', 'name']


class ImageDataSerializer(serializers.ModelSerializer):
    tags = serializers.CharField(write_only=True)  # Comma-separated input
    image_tags = TagSerializer(many=True, read_only=True)

    class Meta:
        model = ImageData
        fields = ['id', 'image', 'image_text', 'image_embedding', 'image_meta', 'image_tags', 'tags']

    def create(self, validated_data):
        tags_str = validated_data.pop('tags', '')
        image_file = validated_data['image']

        # Generate derived data
        embedding = generate_image_embedding(image_file)
        caption = generate_image_text(image_file)
        meta = extract_image_metadata(image_file)

        instance = ImageData.objects.create(
            image=validated_data['image'],
            image_text=caption,
            image_embedding=embedding,
            image_meta=meta
        )

        # Handle tags
        tags = parse_tags(tags_str)
        for tag in tags:
            tag_obj, created = Tag.objects.get_or_create(name=tag)
            instance.image_tags.add(tag_obj)

        return instance