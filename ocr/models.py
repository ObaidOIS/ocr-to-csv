from django.db import models

# Create your models here.


class Ocr_image(models.Model):
    image = models.ImageField(upload_to="static/images/")
    file = models.FileField(upload_to="static/files/")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_filename(self):
        return self.file.name.split('/')[-1]
