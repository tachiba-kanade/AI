from django.urls import path, include
from rest_framework import routers
from search.v1.views import ImageDataViewset, search_images, text_search, upload_and_search, color_search, input_detect

router = routers.SimpleRouter()
router.register(r'image_data', ImageDataViewset)

urlpatterns = [
    path(route='search/', view=search_images),
    path(route='upload_and_search/', view=upload_and_search),
    path(route='text_search/', view=text_search),
    path(route='color_search/', view=color_search),
    path(route='input_detect/',view=input_detect)
] + router.urls