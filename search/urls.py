from django.contrib import admin
from django.urls import path, include
from rest_framework import routers

from search.v1.views import ImageDataViewset

router = routers.SimpleRouter()

router.register(r'get_image_data', ImageDataViewset)

urlpatterns = [] + router.urls
