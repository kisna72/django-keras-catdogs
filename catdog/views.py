from django.shortcuts import render
import keras
from django.conf import settings
from PIL import Image
import numpy 
import keras.backend as K


def convert_to_ndarry(cur_list):
    """Takes a list of ndarrys for images, and converts to a len(list) dimensional ndarray."""
    p = numpy.expand_dims(cur_list[0],0)
    print("Shape " , p.shape)
    for i in range(len(cur_list)):
        nd_item = cur_list[i]
        if (i>0):
            p = numpy.insert(p,-1,cur_list[i],0)
    return p
# Create your views here.
def home(request):
    context = {}
    if (request.method == "GET"):
        pass
    else:
        #Get File and run it through the
        #del(m)
        #m = False
        model = settings.MODEL
        graph = settings.GRAPH
        file_upload = request.FILES['file']
        image = Image.open(file_upload)
        image = image.resize((28,28),Image.ANTIALIAS)
        nd_array = numpy.array(image)
        nd_array_lst = convert_to_ndarry([nd_array,])

        with graph.as_default():
            score = model.predict(nd_array_lst)
            print(score)
            print(score[0])
            if(score[0][0] < 1):
                context["prediction"] = "Cat"
            else:
                context["prediction"] = "Dog"
        #del(m)
        #K.clear_session()

    print(request.method)
   
    return render(request, "index.html", context)