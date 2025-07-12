from django.db import models


class Tag(models.Model):
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.id


class ImageData(models.Model):
    image_text = models.TextField()
    image = models.ImageField(upload_to='image_data/')
    image_embedding = models.BinaryField(blank=True, null=True)
    image_tags = models.ManyToManyField(Tag, related_name="images")
    image_meta = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f" {self.id}"
