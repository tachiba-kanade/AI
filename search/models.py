from django.db import models


class ImageData(models.Model):
    image_text = models.TextField()
    image = models.FileField(upload_to='image_data/')
    image_embedding = models.CharField(max_length=200, blank=True, null=True)
    image_tags = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

def __str__(self):
    return self.id
