import keras as k
import tensorflow
from keras.layers import Dense,Flatten
from keras.models import Sequential
import numpy as np
import os 
import cv2
(X_train,y_train),(X_test,y_test) = k.datasets.mnist.load_data()
print(X_train.shape)
print(X_train[0].shape)
print(y_train)

import matplotlib.pyplot as plt
#plt.imshow(X_train[1],"gray")
#plt.show()

X_train = X_train/255
X_test = X_test/255

#print(X_train[0])

model = Sequential()

model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation="relu"))
model.add(Dense(10,activation = "softmax"))

print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')

history = model.fit(X_train,y_train,epochs=10,validation_split=0.2)

y_prob = model.predict(X_test)
#print(y_prob)

y_pred = y_prob.argmax(axis=1)
print(y_pred)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.show()

#plt.imshow(X_test[1],"gray")
#plt.show()
#pred1 = model.predict(X_test[1].reshape(1,28,28)).argmax(axis=1)
#print(pred1)

img_dir = r"D:\ML\digits"
image_number = 1

while os.path.isfile(os.path.join(img_dir,f"digit{image_number}.png")):
    try:
        img_path = os.path.join(img_dir, f"digit{image_number}.png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Failed to read image{img_path}")
            continue
        
        img = cv2.resize(img, (28, 28))
        img = np.invert(img)
        img = img.reshape(1, 28, 28)

        prediction = model.predict(img)
        print(f"The digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], "gray")
        plt.show()

    except Exception as e:
        print(f"Error processing digit{image_number}.png: {e}")

    finally:
        image_number += 1

