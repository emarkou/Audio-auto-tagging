#additions of code for layer printing

#under line 
#stream_auc = tf.contrib.metrics.streaming_auc(predictions = tf.round(tf.nn.sigmoid(pred)), labels = tf.round(y))
#
#added commands 

image = tf.reshape(x, shape=[-1, 120, 120, 1])

h_conv1=conv2d(image, weights['wc1'], biases['bc1'])
#h_conv1_mp = maxpool2d(h_conv1, k=6)
h_conv2=conv2d(h_conv1, weights['wc2'], biases['bc2'])
#h_conv2_mp = maxpool2d(h_conv2, k=2)
h_conv3 = conv2d(h_conv2, weights['wc3'], biases['bc3'])


#modification of session run so as to acquire variables h_conv1,h_conv2,h_conv3

if step * batch_size % display_step == 0:
            # Calculate batch loss and accuracy for train
            train_loss, train_acc_c, train_acc, train_acc_pc, train_prec, train_rec, train_f1, train_auc, train_stream_auc, train_pred,h_conv1,h_conv2,h_conv3 = sess.run(
                    [cost, accuracy_calc, accuracy, accuracy_per_class,  precision, recall, f1, auc, stream_auc,
                    tf.round(tf.nn.sigmoid(pred)),h_conv1,h_conv2,h_conv3], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
    
#under command 
#test_auc_print = test_auc[0]
#below code is added to save first image of each layer 

            for i in range(0,10):
                img=Image.fromarray(batch_x[i,:,:])
                img.save(str(i)+'init1.tiff')
            for i in range(0,10):
                img_conv1=h_conv1[i,:,:,1].reshape(120,120) #keep only 1 of 128 to show -otherwise use gif image to add all layers in one 
                img_conv1=Image.fromarray(img_conv1)
                img_conv1.save(str(i)+'conv1.tiff')
            for i in range(0,10):
                img_conv2=h_conv2[i,:,:,1].reshape(120,120) #keep only 1 of 284 to show -otherwise use gif image to add all layers in one 
                img_conv2=Image.fromarray(img_conv2)
                img_conv2.save(str(i)+'conv2.tiff')
            for i in range(0,10):
                img_conv3=h_conv3[i,:,:,1].reshape(120,120) #keep only 1 of 768 to show -otherwise use gif image to add all layers in one 
                img_conv3=Image.fromarray(img_conv3)
                img_conv3.save(str(i)+'conv3.tiff')
