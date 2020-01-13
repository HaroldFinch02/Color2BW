import tensorflow as tf
import numpy as np
import cv2
import tflearn
import time
import glob,os
import  image_slicer
from PIL import Image

def sliceImage(filename):
    img = Image.open(filename)
    imgw, imgh = img.size
    originalWidth = imgw
    originalHeight = imgh
    imgw = (int(imgw/128))*128
    imgh = (int(imgh/128))*128
    if imgh < imgw:
        imgw = imgh
    else:
        imgh = imgw
    img = img.resize((imgw, imgh), Image.ANTIALIAS)
    tiles = int(imgw/128)
    tiles = tiles * tiles
    print(tiles)
    img.save('res/res.jpg')
    tiles = image_slicer.slice('res/res.jpg', tiles)
    os.remove('res/res.jpg')
    return tiles, originalWidth, originalHeight

def loadModel():
    # Encoder
    encoder = tflearn.input_data(shape=[None, 128, 128, 3])
    encoder = tflearn.conv_2d(encoder, 64, 3, activation='relu')
    encoder = tflearn.conv_2d(encoder, 32, 3, activation='relu')

    # Decoder
    decoder = tflearn.conv_2d(encoder, 16, 3, activation='relu')
    decoder = tflearn.conv_2d(decoder, 1, 3, activation='relu')

    net = tflearn.regression(decoder, optimizer='adam',
                            learning_rate=0.001, loss='mean_square')


    model = tflearn.DNN(net)
    model.load('SavedModel/bm7420')
    return model

def convertToBW(model, originalWidth, originalHeight):
    for filename in glob.glob('res/*.*'):
        temp=[]
        img = cv2.imread(filename)
        temp.append(np.array(img))
        result = model.predict(np.asarray(temp))
        result = np.reshape(result, (128, 128))
        cv2.imwrite("generatedImages/"+filename,result)

    files = os.listdir('generatedImages/res/')
    files.sort()
    restiles = []
    for filename in files:
        filename = 'res/'+filename
        for tile in tiles:
            if tile.filename == filename:
                tile.image = Image.open('generatedImages/'+filename)
                restiles.append(tile)
                break
    print(originalWidth, originalHeight)            
    res = image_slicer.join(restiles)
    res = res.resize((originalWidth, originalHeight), Image.ANTIALIAS)
    res.save('res.png')
    
    for filename in files:
        os.remove('generatedImages/res/'+filename)
    for filename in glob.glob('res/*.*'):
        os.remove(filename)

if __name__ == "__main__":
    start = time.time()
    filename = str(input("Enter name(or full path if not inside code folder) of file to be converted: "))
    tiles, originalWidth, originalHeight = sliceImage(filename)
    model = loadModel()
    convertToBW(model, originalWidth, originalHeight)
    print("-----------------------------------------------------")
    print("## Execution time(in seconds): ", (time.time()-start))
