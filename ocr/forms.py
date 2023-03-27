from tkinter import Widget
from django import forms
from .models import Ocr_image

class OcrForm(forms.ModelForm):
    class Meta:
        model = Ocr_image
        fields = ['image']
        Widget = {
            'image': forms.FileInput(attrs={'class': 'file-input', 'id': 'image'})
        }