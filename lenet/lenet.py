import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras import backend as K
import numpy

def image_file_to_ndarray(full_path, width=False,height=False):
    """converts single image to ndarray"""
    image = Image.open(full_path)
    if (width and height):#only resize if width and height is passed.
        image = image.resize((width,height), Image.ANTIALIAS)
    nd_array = numpy.array(image)
    return nd_array#of Shape (width,height,channels)

def convert_to_ndarry(cur_list):
    """Takes a list of ndarrys for images, and converts to a len(list) dimensional ndarray."""
    p = numpy.expand_dims(cur_list[0],0)
    print("Shape " , p.shape)
    for i in range(len(cur_list)):
        nd_item = cur_list[i]
        #print(p.shape)
        #print(i)
        if (i>0):
            p = numpy.insert(p,-1,cur_list[i],0)
            #print(p.shape)
    #print(p.shape)
    
    return p

class LeNet():
    def __init__(self,image_depth,image_height,image_channels,x_train,y_train):
        """image_depth = cols of pixels in the image,
        image_height =  rows of  pixels in the image,
        image_channels = number of channels in the image.
        x_train = numpy array of shape (6000,28,28,1). List of 6000 images with 28,28,1 shape.
        y_train = numpy array of shape (6000,10)
        output_classes = y_train.shape(1) ? Yes I suppose.
        """
        self.input_shape = (image_depth,image_height,image_channels)
        self.output_classes = y_train.shape[1]
        #self.x_train = x_train 
        #self.y_train = y_train
        print(self.output_classes)
        print("Input Shape is : ")
        print(self.input_shape)
        
        self.model = Sequential()
        self.model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=self.input_shape )  )
        self.model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2) ))
        self.model.add(Conv2D(filters=50,kernel_size=(2,2),activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(units=500))
        self.model.add(Activation("relu") )
        self.model.add(Dense(units=self.output_classes))
        self.model.add(Activation("softmax"))
        self.model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
    
    def train(self,x_train,y_train,epochs=3,batch_size=None,validation_data=None,validation_split=None,verbose=1):
        print("going to fitting.")
        self.model.fit(x_train,y_train, epochs=epochs,batch_size=batch_size,validation_data=validation_data,
        validation_split=validation_split,verbose=verbose)
        #score=self.model.evaluate(x_test,y_test,verbose=1)
        
    def evaluate(self,x_test, y_test,verbose=1):
        score = self.model.evaluate(x_test,y_test,verbose)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score

    def predict(self, x_test):
        score = self.model.predict(x_test)
        return score 