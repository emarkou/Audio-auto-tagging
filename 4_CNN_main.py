from __future__ import print_function
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.tensorboard.tensorboard
import matplotlib.pyplot as plt

import argparse
import sys

#==============================================================================
#          Define batch load functions
#==============================================================================
def batch_load(files_path, files_list, offset_index, batch_size):

    print('Total number of files in dir: ',len(files_list))

    if offset_index + batch_size < len(files_list):
        print('still files to go')
        batch_x = []
        batch_y = []

        for npy in files_list[offset_index:offset_index+batch_size]: 
            batch_x_instance = np.load(files_path + npy)
            batch_y_instance = np.asarray(list(npy[:50]), dtype = int)
            
            batch_x.append(batch_x_instance)
            batch_y.append(batch_y_instance)

        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(batch_y)
        offset_index = offset_index + batch_size

        print(batch_x.shape)
        print(batch_y.shape)
        print(offset_index)
    
    else:
        print('last batch for current pass')
        batch_x = []
        batch_y = []

        for npy in files_list[offset_index:offset_index+batch_size]: 
            batch_x_instance = np.load(files_path + npy)
            batch_y_instance = np.asarray(list(npy[:50]), dtype = int) 
            
            batch_x.append(batch_x_instance)
            batch_y.append(batch_y_instance)

        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(batch_y)
        offset_index = 0

        print(batch_x.shape)
        print(batch_y.shape)
        print(offset_index)
    
    return batch_x, batch_y, offset_index, len(files_list)


def test_load(files_path, files_list):

    print('Total number of files in dir: ',len(files_list))

    test_x = []
    test_y = []
    test_instances = []

    for npy in files_list: 
        test_instance_name = npy
        test_x_instance = np.load(files_path + npy)
        test_y_instance = np.asarray(list(npy[:50]), dtype = int) 
        
        test_x.append(test_x_instance)
        test_y.append(test_y_instance)
        test_instances.append(test_instance_name)

    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)
    print(test_x.shape)
    print(test_y.shape)
         
    return test_x, test_y, test_instances

#==============================================================================
#       Define all session functions
#==============================================================================
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 120, 120, 1])
    
    # 1st Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=6)

    # 2nd Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    
    # 3rd Convolution Layer (without maxpooling)
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    #conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])

    # conv3shape = conv3.get_shape().as_list()
    # print (conv3shape)
    #fc1 = tf.reshape(conv3, [-1, conv3shape[1] * conv3shape[2] * conv3shape[3]])
    print (fc1.get_shape().as_list())
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    print (fc1.get_shape().as_list())
    fc1 = tf.nn.relu(fc1)
    print (fc1.get_shape().as_list())
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

#==============================================================================
#           Setting CNN layers
#==============================================================================
# Network Parameters
n_input_x1 = 120 # Spectrogram size (120, 120)
n_input_x2 = 120 # Spectrogram size (120, 120)
n_classes  = 50  # magnatagatune number of labels: 50


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input_x1, n_input_x2])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# define layers' weights and biases
weights = {
    # 10x10 conv, 1 input, 64 outputs
    # (filter width, filter depth, number of channels, number of outputs)
    'wc1': tf.Variable(tf.random_normal([10, 10, 1, 64])),
    
    # 10x10 conv, 64 inputs, 128 outputs
    'wc2': tf.Variable(tf.random_normal([10, 10, 64, 128])),
  
    # 10x10 conv, 128 inputs, 256 outputs
    'wc3': tf.Variable(tf.random_normal([10, 10, 128, 256])),
    
   
    # fully connected, 10*10*256 inputs (conv3 output without maxpooling), 1024 outputs
    'wd1': tf.Variable(tf.random_normal([10*10*256, 1024])),
    # 1024 inputs, 50 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#==============================================================================
#           Set all input parameter
#==============================================================================

# define directory where spectrogram train npy's are stored
directory = '<magnatagatune file directory>/npy_files/'

# define directory where spectrogram test npy's are stored
directory_test = '<magnatagatune file directory>/npy_files_test/'

# define path to csv with reduced labels
path_to_labels = '<magnatagatune file directory>/annotations_final_new.csv'

# define path to save plots
directory_plots = '<magnatagatune file directory>/plots/'

# Get all list of files within path
filenames = [x for x in os.listdir(directory)]

# Get all list of files within path
filenames_test = [x for x in os.listdir(directory_test)]

# Set learning Parameters
learning_rate = 0.001
training_iters = 147000
batch_size = 700
offset = 0 
display_step = 700
dropout = 0.75 # Dropout, probability to keep units

#==============================================================================
#           Construct model & Launch session
#==============================================================================

# Construct model
pred = conv_net(x, weights, biases, keep_prob)


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.round(tf.nn.sigmoid(pred)), tf.round(y))

# Mean accuracy over all labels
accuracy_calc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
accuracy = tf.metrics.accuracy(predictions = tf.round(tf.nn.sigmoid(pred)) , labels = y)
accuracy_per_class = tf.metrics.mean_per_class_accuracy(predictions = tf.round(tf.nn.sigmoid(pred)) , labels = y, num_classes = n_classes)

# Precision & recallif step * batch_size == training_size:

#TP = tf.count_nonzero(tf.rdirectory_plotsound(tf.nn.sigmoid(pred)) * tf.round(y))
#TN = tf.count_nonzero((1 - tf.round(tf.nn.sigmoid(pred))) * (1 - tf.round(y)))
#FP = tf.count_nonzero(tf.round(tf.nn.sigmoid(pred)) * (1 - tf.round(y)))
#FN = tf.count_nonzero((1 - tf.round(tf.nn.sigmoid(pred))) * tf.round(y))

precision = tf.metrics.precision(predictions = tf.round(tf.nn.sigmoid(pred)) ,labels = tf.round(y))
recall = tf.metrics.recall(predictions = tf.round(tf.nn.sigmoid(pred)) , labels = tf.round(y))

#precision = TP / (TP + FP)
#recall = TP / (TP + FN)
f1 = 2 * precision[0] * recall[0] / (precision[0] + recall[0])

# Area under curve

auc = tf.metrics.auc(predictions = tf.nn.sigmoid(pred), labels = y)
stream_auc = tf.contrib.metrics.streaming_auc(predictions = tf.nn.sigmoid(pred), labels = y)

# summary is for tensorboard
summary = tf.summary.merge_all()
# Initializing the variabledirectory_plots
init = tf.global_variables_initializer()
loc = tf.local_variables_initializer()

saver = tf.train.Saver(max_to_keep=1) 


# Launch the graph
with tf.Session() as sess:
    #tensorboard loging file location
    summary_writer = tf.summary.FileWriter('<magnatagatune file directory>/log', sess.graph)
    #initate session
	sess.run(init)
    sess.run(loc)
    step = 1
    
    # load test dataset outside of the loop
    test_batch = test_load(directory_test, filenames_test)
    test_batch_x = test_batch[0]
    test_batch_y = test_batch[1]
    test_instances = test_batch[2]
    
    test_auc_print = float("Inf")
    test_auc_print_prev = float(0)
    
    plot_train_loss = []
    plot_train_acc_c= [] 
    plot_train_acc = []
    plot_train_acc_pc = []
    plot_train_prec = []
    plot_train_rec = []
    plot_train_f1 = [] 
    plot_train_auc = [] 
    plot_train_stream_auc = []
    
    plot_test_acc_c= [] 
    plot_test_acc = []
    plot_test_acc_pc = []
    plot_test_prec = []
    plot_test_rec = []
    plot_test_f1 = [] 
    plot_test_auc = [] 
    plot_test_stream_auc = []
    
    # Keep training until reach max iterations
    while step * batch_size <= training_iters and abs(test_auc_print-test_auc_print_prev) > 0.00000001:
        
        if test_auc_print != float("Inf"):
            test_auc_print_directory_plotsprev = test_auc_print
        
               
        next_batch = batch_load(directory, filenames, offset, batch_size)
        batch_x = next_batch[0]
        batch_y = next_batch[1]
        offset = next_batch[2]
        training_size = next_batch[3]

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        
        if step * batch_size % display_step == 0:
            # Calculate batch loss and accuracy for train
            train_loss, train_acc_c, train_acc, train_acc_pc, train_prec, train_rec, train_f1, train_auc, train_stream_auc, train_pred = sess.run(
                    [cost, accuracy_calc, accuracy, accuracy_per_class,  precision, recall, f1, auc, stream_auc,
                    tf.nn.sigmoid(pred)], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            # Calculate accuracy for test
            test_acc_c, test_acc, test_acc_pc, test_prec, test_rec, test_f1, test_auc, test_stream_auc, test_pred = sess.run(
                    [accuracy_calc, accuracy, accuracy_per_class,  precision, recall, f1, auc, stream_auc,
                     tf.nn.sigmoid(pred)], feed_dict={x: test_batch_x, y: test_batch_y, keep_prob: 1.})
            
            test_auc_print = test_auc[0]
            print(
                "Iter " + str(step*batch_size) + "\n"+
                "Minibatch Loss= " + "{:.6f}".format(train_loss) + "\n" +
                
                "Training Accuracy (calculated) = " + "{:.5f}".format(train_acc_c) + "\n"+
                "Training Accuracy  = " + "{:.5f}".format(train_acc[0]) + "\n"+
                "Training Accuracy (per class) = " + "{:.5f}".format(train_acc_pc[0]) + "\n"+
                "Training Precision = " + "{:.5f}".format(train_prec[0]) + "\n"+
                "Training Recall = " + "{:.5f}".format(train_rec[0]) + "\n"+
                "Training F1 = " + "{:.5f}".format(train_f1) + "\n"+
                "Training AUC = " + "{:.5f}".format(train_auc[0]) + "\n"+
                "Training AUC (stream) = " + "{:.5f}".format(train_stream_auc[0]) + "\n"+
                
                "Test Accuracy (calculated) = " + "{:.5f}".format(test_acc_c) + "\n"+
                "Test Accuracy  = " + "{:.5f}".format(test_acc[0]) + "\n"+
                "Test Accuracy (per class) = " + "{:.5f}".format(test_acc_pc[0]) + "\n"+
                "Test Precision = " + "{:.5f}".format(test_prec[0]) + "\n"+
                "Test Recall = " + "{:.5f}".format(test_rec[0]) + "\n"+
                "Test F1 = " + "{:.5f}".format(test_f1) + "\n"+
                "Test AUC = " + "{:.5f}".format(test_auc[0]) + "\n"+
                "Test AUC (stream) = " + "{:.5f}".format(test_stream_auc[0]) + "\n"

            )
        
        plot_train_loss.append((step*batch_size, train_loss))
        plot_train_acc_c.append((step*batch_size, train_acc_c)) 
        plot_train_acc.append((step*batch_size, train_acc[0]))
        plot_train_acc_pc.append((step*batch_size, train_acc_pc[0]))
        plot_train_prec.append((step*batch_size, train_prec[0]))
        plot_train_rec.append((step*batch_size, train_rec[0]))
        plot_train_f1.append((step*batch_size, train_f1)) 
        plot_train_auc.append((step*batch_size, train_auc[0])) 
        plot_train_stream_auc.append((step*batch_size, train_stream_auc[0]))
        
        plot_test_acc_c.append((step*batch_size, test_acc_c)) 
        plot_test_acc.append((step*batch_size, test_acc[0]))
        plot_test_acc_pc.append((step*batch_size, test_acc_pc[0]))
        plot_test_prec.append((step*batch_size, test_prec[0]))
        plot_test_rec.append((step*batch_size, test_rec[0]))
        plot_test_f1.append((step*batch_size, test_f1)) 
        plot_test_auc.append((step*batch_size, test_auc[0])) 
        plot_test_stream_auc.append((step*batch_size, test_stream_auc[0]))
        
        
        if step * batch_size == training_size:
            filenames = random.sample(filenames, len(filenames))
            print('Training list shuffled!\n')
        
        step += 1
          
    print("Optimization Finished!")
    savePath = saver.save(sess, '<magnatagatune file directory>/final_model.ckpt')

#==============================================================================
#           Plotting of Evaluation Metrics
#==============================================================================

a1 = plt.figure(1)
plt.title('Train loss')   
plt.plot(*zip(*plot_train_loss))
a1.show()
a1.savefig(directory_plots + '1')

a2 = plt.figure(2)
plt.title('Train accuracy (calculated)')   
plt.plot(*zip(*plot_train_acc_c))
a2.show()
a2.savefig(directory_plots + '2')


a3 = plt.figure(3)
plt.title('Train accuracy')   
plt.plot(*zip(*plot_train_acc))
a3.show()
a3.savefig(directory_plots + '3')


a16 = plt.figure(16)
plt.title('Train accuracy (per class)')   
plt.plot(*zip(*plot_train_acc_pc))
a16.show()
a16.savefig(directory_plots + '16')


a4 = plt.figure(4)
plt.title('Train precision')
plt.plot(*zip(*plot_train_prec))
a4.show()
a4.savefig(directory_plots + '4')


a5 = plt.figure(5)
plt.title('Train recall')
plt.plot(*zip(*plot_train_rec))
a5.show()
a5.savefig(directory_plots + '5')


a6 = plt.figure(6)
plt.title('Train F1')
plt.plot(*zip(*plot_train_f1))
a6.show()
a6.savefig(directory_plots + '6')


a7 = plt.figure(7)
plt.title('Train AUC')
plt.plot(*zip(*plot_train_auc))
a7.show()
a7.savefig(directory_plots + '7')


a8 = plt.figure(8)
plt.title('Train AUC (streaming)')
plt.plot(*zip(*plot_train_stream_auc))
a8.show()
a8.savefig(directory_plots + '8')


a9 = plt.figure(9)
plt.title('Test accuracy (calculated)')
plt.plot(*zip(*plot_test_acc_c))
a9.show()
a9.savefig(directory_plots + '9')


a10 = plt.figure(10)
plt.title('Test accuracy')
plt.plot(*zip(*plot_test_acc))
a10.show()
a10.savefig(directory_plots + '10')


a17 = plt.figure(17)
plt.title('Test accuracy (per class)')   
plt.plot(*zip(*plot_test_acc_pc))
a17.show()
a17.savefig(directory_plots + '17')


a11 = plt.figure(11)
plt.title('Test precision')
plt.plot(*zip(*plot_test_prec))
a11.show()
a11.savefig(directory_plots + '11')


a12 = plt.figure(12)
plt.title('Test recall')
plt.plot(*zip(*plot_test_rec))
a12.show()
a12.savefig(directory_plots + '12')


a13 = plt.figure(13)
plt.title('Test F1')
plt.plot(*zip(*plot_test_f1))
a13.show()
a13.savefig(directory_plots + '13')


a14 = plt.figure(14)
plt.title('Test AUC')
plt.plot(*zip(*plot_test_auc))
a14.show()
a14.savefig(directory_plots + '14')


a15 = plt.figure(15)
plt.title('Test AUC (streaming)')
plt.plot(*zip(*plot_test_stream_auc))
a15.show()
a15.savefig(directory_plots + '15')

#==============================================================================
#           Sample Extraction from Test
#==============================================================================
 

i = 21
print('Test audio file name: ', test_instances[i])
c = np.vstack((test_pred[[i,]],test_batch_y[[i,]]))
labels = pd.read_csv(path_to_labels, sep=",")
labels = list(labels.columns.values)
labels = labels[1:51]
df = pd.DataFrame (data = c, columns = labels)
df.to_csv(directory_plots + test_instances[i], sep= ',')


#==============================================================================
#           Generation of Confusion Matrices per label for Test Data Set
#==============================================================================
 

c_length = len(test_instances)
c_preds = []

for i in range(0,c_length):
    print('Test audio file name: ', test_instances[i])
    c_preds_instance = test_pred[i,]
            
    c_preds.append(c_preds_instance)


c_preds = np.asarray(c_preds)

#actual predicted + 
#0       2*0     0 -- True Negative 
#1       2*1     3 -- True Positive 
#1       2*0     1 -- False Negative 
#0       2*1     2 -- False Positive

c_final = pd.DataFrame(data = 2 * c_preds + test_batch_y, columns = labels)
c_final = c_final.apply(pd.Series.value_counts).transpose().fillna(0)
c_final.columns= ['TN', 'FN', 'FP', 'TP']
c_final['sum']= c_final['TP']+c_final['FN']
c_final = c_final.sort('sum', ascending = 0)
c_final.to_csv(directory_plots + 'confusion_matrices', sep= ',')
