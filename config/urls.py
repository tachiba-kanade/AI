from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
#this is test
urlpatterns = [
    path('searchadmin/', admin.site.urls),
    path('api/', include('search.urls'), name='api'),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)