# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 14:32:55 2016

@author: Sarick
"""
import glob
import random
from shutil import copyfile  
import numpy as np          
import cv2
import cv
import tweepy
#%%
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
participants = glob.glob("C:\\Users\\Sarick\\Desktop\\Metis\\McNulty Data\\Emotion\\*") #Returns a list of all folders with participant numbers

for x in participants:
    part = "%s" %x[-4:] #store current participant number
    for sessions in glob.glob("%s/*" %x): #Store list of sessions for current participant
        for files in glob.glob("%s/*" %sessions):
            current_session = files.split('_')[-3]
            file = open(files, 'r')
            
            emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.
            
            sourcefile_emotion = glob.glob("C:\\Users\\Sarick\\Desktop\\Metis\\McNulty Data\\source_images\\%s\\%s\\*" %(part, current_session))[-1] #get path for last image in sequence, which contains the emotion
            sourcefile_neutral = glob.glob("C:\\Users\\Sarick\\Desktop\\Metis\\McNulty Data\\source_images\\%s\\%s\\*" %(part, current_session))[0] #do same for neutral image
            
            dest_neut = "C:\\Users\\Sarick\\Desktop\\Metis\\McNulty Data\\sorted_set\\neutral\\%s" %sourcefile_neutral[-21:] #Generate path to put neutral image
            dest_emot = "C:\\Users\\Sarick\\Desktop\\Metis\\McNulty Data\\sorted_set\\%s\\%s" %(emotions[emotion], sourcefile_emotion[-21:]) #Do same for emotion containing image
            
            copyfile(sourcefile_neutral, dest_neut) #Copy file
            copyfile(sourcefile_emotion, dest_emot) #Copy file
            
#%%
faceDet = cv2.CascadeClassifier("C:\\Users\\Sarick\\Anaconda2\\Library\\share\\OpenCV\\haarcascades\\haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("C:\\Users\\Sarick\\Anaconda2\\Library\\share\\OpenCV\\haarcascades\\haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("C:\\Users\\Sarick\\Anaconda2\\Library\\share\\OpenCV\\haarcascades\\haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("C:\\Users\\Sarick\\Anaconda2\\Library\\share\\OpenCV\\haarcascades\\haarcascade_frontalface_alt_tree.xml")
#%%
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions
def detect_faces(emotion):
    files = glob.glob("C:\\Users\\Sarick\\Desktop\\Metis\\McNulty Data\\sorted_set\\%s\\*" %emotion) #Get list of all images with emotion

    filenumber = 0
    for f in files:
        frame = cv2.imread(f) #Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        
        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face2) == 1:
            facefeatures == face2
        elif len(face3) == 1:
            facefeatures = face3
        elif len(face4) == 1:
            facefeatures = face4
        else:
            facefeatures = ""
        
        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print "face found in file: %s" %f
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            
            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite("C:\\Users\\Sarick\\Desktop\\Metis\\McNulty Data\\sorted_set\\dataset\\%s\\%s.jpg" %(emotion, filenumber), out) #Write image
            except:
               pass #If error, pass file
        filenumber += 1 #Increment image number

for emotion in emotions: 
    detect_faces(emotion)
    
#%%
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
ffr = cv2.createFisherFaceRecognizer()
data = {}
def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("C:\\Users\\Sarick\\Desktop\\Metis\\McNulty Data\\sorted_set\\dataset\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))
    
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels



training_data, training_labels, prediction_data, prediction_labels = make_sets()

print "training fisher face classifier"
print "size of training set is:", len(training_labels), "images"
ffr.train(training_data, np.asarray(training_labels))

print "predicting classification set"
cnt = 0
correct = 0
incorrect = 0
predlist = []
for image in prediction_data:
    pred, conf = ffr.predict(image)
    predlist.append(ffr.predict(image))
    if pred == prediction_labels[cnt]:
        correct += 1
        cnt += 1
    else:
        incorrect += 1
        cnt += 1
print ((100*correct)/(correct + incorrect))
print predlist
print prediction_labels


#%%
training_data, training_labels, prediction_data, prediction_labels = make_sets()    
print "training fisher face classifier"
print "size of training set is:", len(training_labels), "images"
ffr.train(training_data, np.asarray(training_labels))
#%%
def get_api(cfg):
  auth = tweepy.OAuthHandler(cfg['consumer_key'], cfg['consumer_secret'])
  auth.set_access_token(cfg['access_token'], cfg['access_token_secret'])
  return tweepy.API(auth)

cfg = { 
"consumer_key"        : "ZLrufvTCpbBjcNE1DFoB6hdAZ",
"consumer_secret"     : "c4XhrnZWRXhFOczKnO7wvcrArfHXzeT9feFyl4v7ggd6ORuWqZ",
"access_token"        : "1155502550-zqlLpgXNdxGIlEZm16pFwHgjWd4YO2cSL5VnZnW",
"access_token_secret" : "siBQjeiksq1h4pzIKZYsEOEaWeWXqBikqeNUiLpTO866O" 
}
api = get_api(cfg)
#%%
#import webbrowser
new = 2
import os
os.chdir("C:\\Users\\Sarick\\Documents\\Python Scripts\\")
if __name__ == '__main__':
    list1 = []
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    print "\n\n\npress space to take picture; press ESC to exit"

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(40)
        if key == 27: # exit on ESC
            break
        if key == 32: # press space to save images
            cv.SaveImage("webcam.jpg", cv.fromarray(frame))
            img = cv.LoadImage("webcam.jpg")            
            webcamframe = cv2.imread("webcam.jpg") 
            gray2 = cv2.cvtColor(webcamframe, cv2.COLOR_BGR2GRAY)
            
                    #Detect face using 4 different classifiers
            face = faceDet.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            face2 = faceDet2.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            face3 = faceDet3.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            face4 = faceDet4.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    
            #Go over detected faces, stop at first detected face, return empty if no face.
            if len(face) == 1:
                facefeatures = face
            elif len(face2) == 1:
                facefeatures == face2
            elif len(face3) == 1:
                facefeatures = face3
            elif len(face4) == 1:
                facefeatures = face4
            else:
                facefeatures = ""            
            for (x, y, w, h) in facefeatures:
                gray2 = gray2[y:y+h, x:x+w]
            
            try:
                out = cv2.resize(gray2, (350, 350))
                cv.SaveImage("webcam-m.jpg", cv2.cv.fromarray(gray2))
                #Resize face so all images have same size
            except:
               pass       
            # pred capture
            pred, conf = ffr.predict(out)
            if pred == 0:
                tweet = u"\U0001F610"
            elif pred == 1:
                #url = "http://google.com"
                #webbrowser.open(url, new=new)
                #mad
                tweet = u"\U0001f620"
            elif pred == 2:
                tweet = u"\U0001f644"
            elif pred == 3:
                tweet = u"\U0001F922"
            elif pred == 4:
                tweet = u"\U0001F631"
            elif pred == 5:
                tweet = u"\U0001F61B"
            elif pred == 6:
                tweet = u"\U0001F622"
            elif pred == 7:
                tweet = u"\U0001F62F"            
            for status in tweepy.Cursor(api.user_timeline).items():
                try:
                    api.destroy_status(status.id)                    
                except:
                    pass
            status = api.update_status(status=tweet)
            print pred, conf
            list1.append(pred)
    vc.release()
    cv2.destroyWindow("preview") 
#%%
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
'''
batch_size = 100
nb_classes = 8
nb_epoch = 2
# input image dimensions
img_rows, img_cols = 350, 350
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(img_rows, img_cols, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')


model.fit(train_data, train_target, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_split=0.1)


'''













input_num_units = 122500
hidden_num_units = 15
output_num_units = 8
epochs = 2
batch_size = 100
from keras.models import Sequential
from keras.layers import Dense

# create model
model = Sequential([
  Dense(output_dim=hidden_num_units, input_dim=input_num_units, activation='relu'),
  Dense(output_dim=output_num_units, input_dim=hidden_num_units, activation='softmax'),
])

# compile the model with necessary attributes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

training_data2 = np.stack(training_data)
training_labels2 = np.array(training_labels)
training_labels2 = training_labels2.reshape((-1, 1))
training_labels2 = training_labels2.flatten()
prediction_data2 = np.array(prediction_data)
blah = np.array(training_data)
blah2 = blah.reshape((521, 122500))
model.fit(blah2, training_labels, batch_size=batch_size, nb_epoch=epochs)



          #show_accuracy=True, verbose=1, validation_data=(prediction_data2, prediction_labels))

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
mnist = fetch_mldata("MNIST original")
# rescale the data, use the traditional train/test split
X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]



mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=10, alpha=1e-4,
                    solver='lbfgs', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
mlp.fit(blah2, training_labels)
print("Training set score: %f" % mlp.score(blah2, training_labels))
print("Test set score: %f" % mlp.score(X_test, y_test))

fig, axes = plt.subplots(1, 2)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(350, 350), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

test_data_blah = np.array(prediction_data).reshape(127, 122500)
probability = mlp.predict(test_data_blah)

plt.show()