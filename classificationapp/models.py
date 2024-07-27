from django.db import models

# Create your models here.

class userreg(models.Model):
    username = models.CharField(max_length=20,null=True)
    email = models.CharField(max_length=20,null=True)
    password = models.CharField(max_length=20,null=True)
    
