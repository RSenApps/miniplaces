import os, datetime
import numpy as np
import tensorflow as tf
from DataLoader import *
import resnet
from multiprocessing.pool import ThreadPool

# Dataset Parameters
batch_size = 128
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.1
learning_rate_decay = .8
dropout = 0.5 # Dropout, probability to keep units
training_iters = 5000

step_display = 100
step_save = 1000
path_save = './resnet18/'
weight_decay = .003

momentum = .9

print('learning_rate: ', learning_rate, ' decay: ', learning_rate_decay, ' momentum: ', momentum, ' weight_decay: ', weight_decay)

if not os.path.exists(path_save):
    os.makedirs(path_save)

start_from = '' #'./resnet18/-5000'
start_step = 0 #5000


# Construct dataloader
opt_data_train = {
    'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../../images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../../images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

#loader_train = DataLoaderDisk(**opt_data_train)
#loader_val = DataLoaderDisk(**opt_data_val)
loader_train = DataLoaderH5(**opt_data_train)
loader_val = DataLoaderH5(**opt_data_val)

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
#boundaries = [5000,10000]
#values = [.1,.01,.001]
#learning_rate = tf.train.piecewise_constant(global_step,boundaries,values)
learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                           1000, learning_rate_decay, staircase=True)
train_optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True).minimize(loss, global_step=global_step)
# Evaluate model
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

    step = start_step
    while True:
        training_step = 0
        async_result = pool.apply_async(loader_train.next_batch, [batch_size])
        while training_step < training_iters:
            # Load a batch of training data
            images_batch, labels_batch = async_result.get()  #loader_train.next_batch(batch_size)
            async_result = pool.apply_async(loader_train.next_batch, [batch_size])
            #images_batch, labels_batch = loader_train.next_batch(batch_size)

            if step % step_display == 0:
                print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

                # Calculate batch loss and accuracy on training set
                l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, train_mode: False})
                print("-Iter " + str(step) + ", Training Loss= " + \
                      "{:.4f}".format(l) + ", Accuracy Top1 = " + \
                      "{:.2f}".format(acc1) + ", Top5 = " + \
                      "{:.2f}".format(acc5))

                # Calculate batch loss and accuracy on validation set
                images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)
                l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, train_mode: False})
                print("-Iter " + str(step) + ", Validation Loss= " + \
                      "{:.4f}".format(l) + ", Accuracy Top1 = " + \
                      "{:.2f}".format(acc1) + ", Top5 = " + \
                      "{:.2f}".format(acc5))

            # Run optimization op (backprop)
            sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, train_mode: True})

            step += 1
            training_step += 1
            # Save model
            if step % step_save == 0:
                saver.save(sess, path_save, global_step=step)
                print("Model saved at Iter %d !" %(step))

        print("Optimization Finished!")


        # Evaluate on the whole validation set
        print('Evaluation on the whole validation set...')
        num_batch = loader_val.size()//batch_size
        acc1_total = 0.
        acc5_total = 0.
        loader_val.reset()
        for i in range(num_batch):
            images_batch, labels_batch = loader_val.next_batch(batch_size)
            acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, train_mode: False})
            acc1_total += acc1
            acc5_total += acc5
            print("Validation Accuracy Top1 = " + \
                  "{:.2f}".format(acc1) + ", Top5 = " + \
                  "{:.2f}".format(acc5))

        acc1_total /= num_batch
        acc5_total /= num_batch
        print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))
