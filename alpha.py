import numpy as np
from tqdm import tqdm
from random import shuffle
import os, math, cv2, random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
import tensorflow as tf
from PIL import Image

def make_square(im, min_size=256, fill_color=(255, 255, 255, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, ((size - x) // 2, (size - y) // 2))
    return new_im



def loadData(DIR = './train', limit = 50):
    IMG_SIZE = 64
    data = []
    labels = []
    for i, im in enumerate(os.listdir(DIR)):    
        if not im.startswith('.'):
            path = os.path.join(DIR, im)
            if i >= limit: break
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            img = img/255.0
            data.append([np.array(img)])
    return np.array(data).reshape(-1,IMG_SIZE,IMG_SIZE,3)



def saveGraph(graph):
    with tf.Session(graph=graph) as sess:
        filename = "./summary_log/VS-"+time.strftime("%H%M%S")
        writer = tf.summary.FileWriter(filename, sess.graph)
    print("Tensorboard summary saved to "+filename) 

    
def lrelu(b,alpha=0.1):
    return tf.maximum(alpha*b,b)
    
    
def iterate_minibatches(inputs, batchsize): #Using python generator
    m = inputs.shape[0] 
    indices = np.arange(m)
    np.random.shuffle(indices)
    for index in range(0, m - batchsize + 1, batchsize): # 1 is when SGD
        batch = indices[index:index + batchsize]
        yield inputs[batch]
    if m % batchsize != 0:
        batch = indices[math.floor(m/batchsize)*batchsize:m]
        yield inputs[batch]















