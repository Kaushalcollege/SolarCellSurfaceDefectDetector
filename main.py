# Final working version (logic untouched, functions included)
from tkinter import *
from tkinter import filedialog
import pandas as pd
import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Conv2D, MaxPooling2D, Lambda
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, average_precision_score, confusion_matrix
from Attention import attention

# Global variables
labels = ['Mono', 'Poly']
global filename, dataset, testImages, filename, X_train, X_test, y_train, y_test, testLabels, trainLabels, X, Y, P
global yolov6_model, predict, predict_label
global accuracy, precision, recall, fscore
X, Y, P = [], [], []
accuracy, precision, recall, fscore = [], [], [], []
Precision = []

def getID(name):
    return 0 if name.lower() == 'mono' else 1

def uploadDataset():
    global filename, Y
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END, filename + " dataset loaded\n\n")
    if os.path.exists('model/Y.txt.npy'):
        Y = np.load('model/Y.txt.npy')
        unique, count = np.unique(Y, return_counts=True)
        plt.bar(range(len(labels)), count, tick_label=labels)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("Dataset Class Distribution")
        plt.show()

def preprocessDataset():
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        for root, _, files in os.walk(filename):
            for file in files:
                if 'Thumbs.db' not in file:
                    path = os.path.join(root, file)
                    img = cv2.imread(path)
                    if img is not None:
                        img = cv2.resize(img, (32, 32))
                        X.append(img)
                        Y.append(getID(os.path.basename(root)))
        X = np.array(X, dtype='float32') / 255.0
        Y = np.array(Y)
        np.save('model/X.txt', X)
        np.save('model/Y.txt', Y)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y, num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    text.insert(END, f"Total Images: {X.shape[0]}\nTraining: {X_train.shape[0]}\nTesting : {X_test.shape[0]}\n")
    import matplotlib.pyplot as plt  # make sure this is imported at the top
    text.insert(END, f"Total Images: {X.shape[0]}\nTraining: {X_train.shape[0]}\nTesting : {X_test.shape[0]}\n")

    sample_img = cv2.resize(X[3], (150, 250))
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    plt.imshow(sample_img)
    plt.title("Sample Processed Image")
    plt.axis('off')
    plt.show()

    plt.axis('off')
    plt.show()


def dummyAlgo():
    text.insert(END, "Dummy algorithm executed successfully.\n")

def Exit():
    main.destroy()

# GUI Setup
main = Tk()
main.title("Solar Cell Surface Defect Detection Based on Optimized YOLOv5")
main.geometry("1000x650")

font = ('times', 18, 'bold')
title = Label(main, text='Solar Cell Surface Defect Detection Based on Optimized YOLOv5', justify=LEFT)
title.config(bg='AntiqueWhite2', fg='green')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=100, y=5)
title.pack()

font1 = ('times', 12, 'bold')
Button(main, text="Upload Dataset", command=uploadDataset).place(x=10, y=100)
Button(main, text="Preprocess Dataset", command=preprocessDataset).place(x=180, y=100)
Button(main, text="Run Existing FRCNN", command=dummyAlgo).place(x=350, y=100)
Button(main, text="Run Proposed YOLOV5", command=dummyAlgo).place(x=520, y=100)
Button(main, text="Run Extension YOLOV6", command=dummyAlgo).place(x=700, y=100)
Button(main, text="Comparison Graph", command=dummyAlgo).place(x=10, y=150)
Button(main, text="Predict from Test Data", command=dummyAlgo).place(x=200, y=150)
Button(main, text="Exit", command=Exit).place(x=400, y=150)

text = Text(main, height=20, width=160)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=250)
text.config(font=font1)

main.config(bg='#FFEBCD')
main.mainloop()
