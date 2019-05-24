import warnings
warnings.filterwarnings("ignore")



import tensorflow as tf
import os, cv2, random
import numpy as np
import pandas as pd
import time, math
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import click


@click.command()
@click.option('--dir', default=None, show_default=True, help='Relative path to model.')
@click.option('--minibatch_size', default=256, show_default=True)
@click.option('--epoch', default=5, show_default=True)
@click.option('--learning_rate', default=0.005, show_default=True)
@click.option('--limit', default=9999, show_default=True)
@click.option('--img_size', default=64, show_default=True)
@click.option('--idx', default=4534, show_default=True)
def main(dir, minibatch_size, epoch, learning_rate, limit, img_size, idx):
    graph, filename, opt = model()
    cost = graph.get_tensor_by_name('cost:0')
    # opt = graph.get_tensor_by_name('opt:0')
    X = graph.get_tensor_by_name('input_:0')
    Y = graph.get_tensor_by_name('output_:0')
    lr = graph.get_tensor_by_name('lr_:0')
    encoded = graph.get_tensor_by_name('encoding/encoding/AvgPool:0')

    train = loadData(limit=99999)
    with tf.Session(graph=graph) as sess:
        writer = tf.summary.FileWriter(filename, sess.graph) 
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # saver.restore(sess, tf.train.latest_checkpoint('./saved_model/'))
        for ep in range(epoch):
            avg_cost = 0
            for i, minibatch in enumerate(iterate_minibatches(train, minibatch_size)):
                batch_cost, _ = sess.run([cost, opt], feed_dict={X: minibatch,
                                                                 Y: minibatch,
                                                                   lr: learning_rate})
            print("Epoch: {}/{}...".format(ep+1, epoch), \
                  "Training loss: {:.4f}".format(batch_cost))
        saver.save(sess, "./saved_model/model")


    embeddings = []
    labels = []
    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('./saved_model/'))
        for i, im in (enumerate(os.listdir(dir))):     # tqdm: Professional progress bar 
            if not im.startswith('.'):
                path = os.path.join(dir, im)
                if i >= limit: break
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size,img_size))
                img = img/255.0
                embeddings.append(sess.run(encoded,feed_dict={X:img.reshape((-1,64,64,3))}).reshape(-1,256))
                labels.append(int(path.split('.')[-2].split('-')[-1]))  
    labels = np.array(labels)
    embeddings = np.array(np.squeeze(embeddings))


    data = pd.DataFrame(embeddings)
    data['labels'] = labels
    data.set_index('labels', inplace=True)
    data.sort_index(inplace=True)


    testEmbeddings = getEmbeddings(idx)
    score = getPredictions(testEmbeddings,data)
    images =  getPredictedImages(score)
    im = getTestimage(idx)
    # plotImages(idx, im, images, score)


    '''
    #Calculating score for every item and writing it to dictionary as the given sample submission format.
    I havev made a cutt off score of 0.5; anything below 0.5 doesn't count.
    Only 500 of the duplicates are calculated and stored
    '''
    sc = []
    out = dict()
    for i in range(1):
        testEmbeddings = getEmbeddings(i)
        score = getPredictions(testEmbeddings,data)
        ls = [[getIndex(score.index[i]), score.score.iloc[i]] for i in range(6) if score.score.iloc[i]>0.5 and score.index[i] != i]
        ind, _ = getinfo(i)
        out[str(ind)] = ls  
    print(out)



def model():
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float32,[None,64,64,3], name='input_')
        Y = tf.placeholder(tf.float32,[None,64,64,3], name='output_')    
    with graph.as_default():
        with tf.name_scope('conv'):
            conv0 = tf.layers.conv2d(X,filters=4,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=lrelu,name='conv0')
        with tf.name_scope('pooling'):
            maxpool0 = tf.layers.max_pooling2d(conv0,pool_size=(2,2),strides=(2,2),name='pool0')     
        
        with tf.name_scope('conv'):
            conv1 = tf.layers.conv2d(maxpool0,filters=4,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=lrelu,name='conv1')
        with tf.name_scope('pooling'):
            maxpool1 = tf.layers.max_pooling2d(conv1,pool_size=(2,2),strides=(2,2),name='pool1')
        with tf.name_scope('conv'):
            conv2 = tf.layers.conv2d(maxpool1,filters=8,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=lrelu,name='conv2')
        with tf.name_scope('pooling'):
            maxpool2 = tf.layers.max_pooling2d(conv2,pool_size=(2,2),strides=(2,2),name='pool2')
        with tf.name_scope('conv'):
            conv3 = tf.layers.conv2d(maxpool2,filters=16,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=lrelu,name='conv3')
        with tf.name_scope('encoding'):
            encoded = tf.layers.average_pooling2d(conv3,pool_size=(2,2),strides=(2,2),name='encoding')

    with graph.as_default():
        with tf.name_scope('decoder'):
            upsample1 = tf.layers.conv2d_transpose(encoded,filters=16,kernel_size=3,padding='same',strides=2,name='upsample1')
            conv4 = upsample1 #tf.layers.conv2d(upsample1,filters=16,kernel_size=(3,3),strides=(1,1),padding='SAME',name='conv4',use_bias=True,activation=lrelu)
            upsample2 = tf.layers.conv2d_transpose(conv4,filters=8,kernel_size=3,padding='same',strides=2,name='upsample2')
            conv5 = upsample2 #tf.layers.conv2d(upsample2,filters=8,kernel_size=(3,3),strides=(1,1),name='conv5',padding='SAME',use_bias=True,activation=lrelu)
            upsample3 = tf.layers.conv2d_transpose(conv5,filters=8,kernel_size=5,padding='same',strides=2,name='upsample3')
            conv6 = tf.layers.conv2d(upsample3,filters=4,kernel_size=(5,5),strides=(1,1),name='conv6',padding='SAME',use_bias=True,activation=lrelu)
            
            upsample4 = tf.layers.conv2d_transpose(conv6,filters=8,kernel_size=5,padding='same',strides=2,name='upsample4')
            conv7 = tf.layers.conv2d(upsample4,filters=4,kernel_size=(5,5),strides=(1,1),name='conv7',padding='SAME',use_bias=True,activation=lrelu)
            logits = tf.layers.conv2d(conv7,filters=3,kernel_size=(3,3),strides=(1,1),name='logits',padding='SAME',use_bias=True)
            decoded = tf.sigmoid(logits,name='recon')

    with graph.as_default():
        '''
        Defining loss function and optimizer
        '''
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=Y)
        lr = tf.placeholder(tf.float32, shape=[], name='lr_')
        cost = tf.reduce_mean(loss, name='cost')
        opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost, name='opt') #optimizer
        
        summaryMerged = tf.summary.merge_all() #For tensorboard
        filename="./summary_log/run-"+time.strftime("%d%m-%H%M%S")
        return graph, filename, opt


def getEmbeddings(idx):
    '''
    Calculates embedding vector given a single image
    input: index of single image to be tested
    return: Embeddings of shape (256,)
    '''
    IMG_SIZE = 64


    graph, filename, opt = model()
    cost = graph.get_tensor_by_name('cost:0')
    # opt = graph.get_tensor_by_name('opt:0')
    X = graph.get_tensor_by_name('input_:0')
    Y = graph.get_tensor_by_name('output_:0')
    lr = graph.get_tensor_by_name('lr_:0')
    encoded = graph.get_tensor_by_name('encoding/encoding/AvgPool:0')

    
    img = cv2.imread('./train/im-400-{}.jpg'.format(idx))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    img = img/255.0
    
    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('./saved_model/'))
        dist = sess.run(encoded,feed_dict={X:img.reshape((-1,64,64,3))})
    
    return np.squeeze(dist.reshape(-1,4*4*16))




#Reading images with index of 'scores' dataframe
def getPredictedImages(score):
    '''
    loading images with id equals to index of 'score'
    '''
    images = []
    IMG_SIZE = 200
    for idx in score.index:
        path = './train/im-400-{}.jpg'.format(idx)
        img = Image.open(path)
        img  = make_square(img)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        images.append([np.array(img)])
    return np.array(images).reshape(-1, 200,200,3)




def getPredictions(testEmbeddings,data):
    '''
    input: Embeddings a single test image; shape- (256,)
    return: Dataframe contains details of top 10 similar images
    and 
    '''


    dist = (np.array(data) - testEmbeddings)**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)    
    '''
    dist : Euclidean distance between testEmbedding and embeddings of all the images. dtype: numpy array
    '''
    
    df = pd.DataFrame({'distance':dist})
    df.index = data.index
    df.sort_values('distance', ascending=True, inplace=True) 
    score = df[:10] #Taking highest 10 
    score['score'] = df.distance[:10].apply(lambda x: np.round(1-np.tanh(x)**10, 3)).values
    '''
    tanh is used to map distance to 0 to 1
    '''
    return score



def getTestimage(idx):
    '''
    This is for ploting the original image
    reads image from directory given index
    '''
    path = './train/im-400-{}.jpg'.format(idx)
    img = Image.open(path)
    img  = make_square(img)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img





def plotImages(idx, im, images, score):
    '''
    Function to plot original image and images with highest scores; 
    plots 10 images with xlabels: Score, original id and seller-name 
    '''
    fig, axes = plt.subplots(5,2, figsize = (15,25))
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
    
    for i, ax in enumerate(axes.flat):
        if i==0:
            idX, sellerName = getinfo(idx)
            ax.imshow(im, cmap = 'binary')
            xlabel = "Original image \n id: {} \n Seller: {}".format(idX, sellerName)
            ax.set_xlabel(xlabel, fontsize = 13)
        else:    
            ax.imshow(images[i-1], cmap = 'binary')
            
            idX, sellerName = getinfo(score.index[i])
            xlabel = "score: {} \nid:{} \n Seller: {}".format(score.score.iloc[i-1], idX, sellerName)
        
            ax.set_xlabel(xlabel, fontsize = 12)
        
        #Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()

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




def create_placeholders(n_H0, n_W0, n_C0, n_y): 
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    return X, Y

def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        with tf.name_scope("relu"):
            act = tf.nn.relu(conv + b)
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
        with tf.name_scope("max_pool"):
            max_pool = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    return max_pool


def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

    
def compute_cost(Z, Y):
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))
        tf.summary.scalar("cost", cost)
    return cost



def getIndex(idx):
    Data = pd.read_csv('./Data/data.csv')
    tunicSlice = pd.read_csv('./Data/tunics.csv')
    '''
    index of subset and original data is different.
    This function will return original index given the subset index
    '''
    pid = Data[Data.index==idx]['productId'].values[0]
    idxTunic = tunicSlice[tunicSlice.productId==pid]['index'].values[0]
    return idxTunic

def getinfo(idx):
    Data = pd.read_csv('./Data/data.csv')
    tunicSlice = pd.read_csv('./Data/tunics.csv')
    pid = Data[Data.index==idx]['productId'].values[0]
    idxTunic = tunicSlice[tunicSlice.productId==pid]['index'].values[0]
    return idxTunic, tunicSlice[tunicSlice['index']==idxTunic]['sellerName'].values[0]






if __name__ == '__main__':
	main()