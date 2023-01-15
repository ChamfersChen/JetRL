"""jetimg URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

from imgprocess import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("index/", views.index),
    path("index2/",views.index2),
    path("pic_handle/", views.pic_handle),
    path("show_result_18/", views.show_result_18),
    path("show_result_50/", views.show_result_50),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
