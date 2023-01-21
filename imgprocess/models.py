<<<<<<< HEAD
from django.db import models

class Image(models.Model):

    photo = models.ImageField(null=True, blank=True)

    def __str__(self):
=======
from django.db import models

class Image(models.Model):

    photo = models.ImageField(null=True, blank=True)

    def __str__(self):
>>>>>>> 2f0eaee8a477a91af37b042851c4e144fb1e6671
        return self.photo.name