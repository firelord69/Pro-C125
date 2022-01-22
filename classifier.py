import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X,y = fetch_openml('mnist_784', version=1, return_X_y=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42,train_size=7500,test_size=2500)

X_train_scaled = X_train/255
X_test_scaled = X_test/255

clf = LogisticRegression(slover='saga',multi_class='multinomial').fit(X_train_scaled,y_train)

def get_predicition(image):
    im_pil = Image.open(image)
    imagebw = im_pil.convert('L')
    imagebwresized = imagebw.resize((28,28), Image.ANTIALIAS)
    pixelfilter = 20
    minpixel = np.percentile(imagebwresized, pixelfilter)
    imagebwresizedinvertedscaled = np.clip(imagebwresized-minpixel, 0, 255)
    max_pixel = np.max(imagebwresized)
    imagebwresizedinvertedscaled = np.asarray(imagebwresizedinvertedscaled)/max_pixel
    test_sample = np.array(imagebwresizedinvertedscaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]