from django.db import models
import os


class Tag(models.Model):
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.id}"


class ImageData(models.Model):
    image_text = models.TextField()
    image = models.ImageField(upload_to='image_data/')
    image_embedding = models.BinaryField(blank=True, null=True)
    image_tags = models.ManyToManyField(Tag, related_name="images")
    image_meta = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    dominant_color = models.CharField(max_length=20, blank=True, null=True)

    def delete(self, using=None, keep_parents=False):
        if self.image and os.path.isfile(self.image.path):
            os.remove(self.image.path)
        return super().delete(using=using, keep_parents=keep_parents)

    def __str__(self):
        return f" {self.id} "
