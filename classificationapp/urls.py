
from django.urls import path
from .import views

urlpatterns = [
       path('',views.home,name='home'),
       path('register/',views.register,name='register'),
       path('login/',views.login,name='login'),
    path('log/',views.log,name='log'),
   path('upload/', views.blood_cell_detection, name='upload'),
   path('result/',views.processed_image,name='result')   #  path('clsy/',views.classify_image,name='clsy'),
   #   path('classify/', views.classify_image1, name='classify_image'),

    
]
