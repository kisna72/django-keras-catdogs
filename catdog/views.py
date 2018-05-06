from django.shortcuts import render
import keras
from django.conf import settings
import matplotlib.pyplot as plt

model_file = settings.STATIC_ROOT + "/catdog/my_model.h5"
m = keras.models.load_model(model_file)
# Create your views here.
def home(request):
    
    if (request.method == "GET"):
        pass
    else:
        #Get File and run it through the d
        print(request.FILES)
        print(request.POST['file'])
        file_upload = request.POST['file'][0]
        print("Uploaded")
        print(type(file_upload))
        d = file_upload.getdata()
        print(d)
        arr = plt.imread(file_upload)
        print(arr.shape)
    print(request.method)
    return render(request, "index.html")