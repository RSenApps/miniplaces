import os, datetime
import numpy as np
import tensorflow as tf
from DataLoader import *
import resnet
from multiprocessing.pool import ThreadPool

# Dataset Parameters
batch_size = 1
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.002
dropout = 0.8 # Dropout, probability to keep units
training_iters = 500
step_display = 100
step_save = 1000
path_save = './resnet18/'
weight_decay = .002
momentum = .9

if not os.path.exists(path_save):
    os.makedirs(path_save)

start_from = './resnet18-Friday-830/-15000'
start_step = 0 #5000


# Construct dataloader
opt_data_train = {
    'data_h5': 'miniplaces_256_test.h5',
    'data_root': '../../../images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/test.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    'data_h5': 'miniplaces_256_test.h5',
    'data_root': '../../../images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/test.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
#keep_dropout = tf.placeholder(tf.float32)
train_mode = tf.placeholder(tf.bool)

# Construct model
res = resnet.ResNet()
logits = res.build_tower(x,train_mode)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
#cost = -tf.reduce_sum(y*tf.log(logits))

regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
loss += reg_term

#train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
global_step = tf.Variable(0,trainable=False)
boundaries = [10000,20000]
values = [.1,.01,.001]
learning_rate = tf.train.piecewise_constant(global_step,boundaries,values)
train_optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True).minimize(loss, global_step=global_step)
# Evaluate model
top5 = tf.nn.top_k(logits,k=5,sorted=True)
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())
print('working')
pool = ThreadPool(processes=1)

# Launch the graph
with tf.Session() as sess:
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)

    # Evaluate on the whole test set
    print('Evaluation on the whole test set...')
    num_batch = loader_val.size()//batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    for i in range(num_batch):
        images_batch, labels_batch = loader_val.next_batch(batch_size)
        t5 = sess.run([top5], feed_dict={x: images_batch, train_mode: False})
        print("test/" + str(i+1).zfill(8) + ".jpg " + " ".join(str(x) for x in t5[0].indices[0]))
