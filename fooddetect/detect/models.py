from django.db import models
from django.contrib.postgres.fields import ArrayField


class Standard(models.Model):
    class_number = models.IntegerField(unique=True)
    class_name = models.CharField(max_length=255)
    temperature = models.FloatField()
    weight = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to="standard/")
    embedding = ArrayField(models.FloatField())

    def __str__(self):
        return self.class_name
