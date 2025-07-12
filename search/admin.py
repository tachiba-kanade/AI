from django.contrib import admin
from .models import ImageData, Tag

# Register your models here.

@admin.register(ImageData)
class ImageDataAdmin(admin.ModelAdmin):
    list_display = ('image_text', )

@admin.register(Tag)
class ImageDataAdmin(admin.ModelAdmin):
    list_display = ('name', )