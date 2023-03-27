from django.http import HttpResponse
from django.views.generic import  CreateView, View
from .forms import OcrForm
from .models import Ocr_image
from .convert2csv import convert
import cv2
import numpy as np
from django.shortcuts import redirect 
import time

# Create your views here.

class UploadOcr(CreateView):
    template_name = "home.html"
    form_class = OcrForm
    success_url = "ocr/download/"
    in_image = None
    def form_valid(self, form):
        img = cv2.imdecode(np.fromstring(form.cleaned_data['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        in_image = form.cleaned_data['image']
        print('in_image')
        if in_image:
            print('inside_image')
            df = convert(img)
            print('df')
            time_stamp = "-"+ time.strftime("%Y%m%d-%H%M%S")
            df.to_csv("static/files/"+ in_image.name.split('.')[0]+ time_stamp + '.csv', index=False)
            ocr_obj = Ocr_image.objects.create(image=in_image, file='static/files/'+in_image.name.split('.')[0]+ time_stamp   + '.csv')
            ocr_obj.save()
            print('ocr_obj')
            return redirect('download', fileID=ocr_obj.id)
        return super().form_valid(form)

class DownloadView(View):
    print("Hello World")
    def get(self, request, *args, **kwargs):
        files = Ocr_image.objects.get(id=self.kwargs['fileID'])
        filename = files.get_filename()
        url = "static/files/" + filename
        download_content = open(url, 'rb')
        if url:
            response = HttpResponse(download_content,content_type="text/csv")
            response['Content-Disposition'] = f'attachment; filename={filename}'
            return response
        return HttpResponse("File not found")
