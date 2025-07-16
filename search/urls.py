from django.urls import path, include
from rest_framework import routers
from search.v1.views import ImageDataViewset, search_images, upload_and_search, text_search

router = routers.SimpleRouter()
router.register(r'image_data', ImageDataViewset)

urlpatterns = [
    path('search/', search_images),
    path('upload_and_search/', upload_and_search),
    path(route='text_search/',view=text_search),
] + router.urls