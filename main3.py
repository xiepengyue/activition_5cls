# -*- encoding: utf8 -*-
# author: ronniecao
import sys
import os
import numpy
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorboardX import SummaryWriter

from src.layer.basicblock import ConvLayer
from src.layer.dense_layer import DenseLayer
from src.layer.pool_layer import PoolLayer
import Tfrecord
import image_data_loader

from utils.utils import progress_bar
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, accuracy_score
import numpy as np

import argparse



class ConvNet():

    def __init__(self, n_channel=3, n_classes=5, image_size=128, n_layers=5):
        # 设置超参数
        self.n_channel = n_channel
        self.n_classes = n_classes
        self.image_size = image_size
        self.n_layers = n_layers



        # 输入变量
        self.images = tf.placeholder(dtype=tf.float32, shape=[None, self.image_size, self.image_size, self.n_channel],name='images')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.n_classes], name='labels')  ##############
        #self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.BN_training = tf.placeholder(dtype=tf.bool, name='BN_mode')
        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step')


        # 网络输出
        self.logits = self.inference(self.images, is_training=self.BN_training)

        # 目标函数
        #cls_weight = [0.150, 0.075, 0.649, 0.072, 0.054]   #104w
        cls_weight = [0.177, 0.078, 0.607, 0.077, 0.061]  #121w
        #{'listen': 0.177, 'write': 0.078, 'handup': 0.607, 'positive': 0.077, 'negative': 0.061}  121w
        self.weight_label=tf.stack([self.labels[:,0]*cls_weight[0],
                                   self.labels[:,1]*cls_weight[1],
                                   self.labels[:,2]*cls_weight[2],
                                   self.labels[:,3]*cls_weight[3],
                                   self.labels[:,4]*cls_weight[4]],axis=-1)
        #label_w = self.labels * weight_label
        self.objective = -tf.reduce_sum(self.weight_label * tf.nn.log_softmax(self.logits))

        '''
        self.objective = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels))
        '''
        # weights =
        # self.objective = tf.reduce_sum(
        #     tf.losses.sparse_softmax_cross_entropy(
        #         logits=self.logits, labels=self.labels))

        tf.add_to_collection('losses', self.objective)
        self.avg_loss = tf.add_n(tf.get_collection('losses'))
        # 优化器
        '''
        lr = tf.cond(tf.less(self.global_step, 50000),
                     lambda: tf.constant(0.01),
                     lambda: tf.cond(tf.less(self.global_step, 100000),
                                     lambda: tf.constant(0.005),
                                     lambda: tf.cond(tf.less(self.global_step, 150000),
                                                     lambda: tf.constant(0.0025),
                                                     lambda: tf.constant(0.001))))
        '''
        lr = tf.cond(tf.less(self.global_step, 100000),
                     lambda: tf.constant(0.01),
                     lambda: tf.cond(tf.less(self.global_step, 200000),
                                     lambda: tf.constant(0.005),
                                     lambda: tf.cond(tf.less(self.global_step, 250000),
                                                     lambda: tf.constant(0.0025),
                                                     lambda: tf.constant(0.001))))



        #self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.avg_loss, global_step=self.global_step)

        self.optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr)#.minimize(self.avg_loss, global_step=self.global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.avg_loss, global_step=self.global_step)

            
        # 观察值
        self.pred = tf.argmax(self.logits, 1)

        correct_prediction = tf.equal(tf.argmax(self.labels, 1), self.pred)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


    def inference(self, images, is_training=True):

        conv_layer0_list = ConvLayer(
                input_shape=(None, self.image_size+6, self.image_size+6, self.n_channel),
                n_size=7, n_filter=64, stride=2, activation='relu',
                #batch_normal=True, weight_decay=1e-4, name='conv0', padding='VALID'))
                batch_normal=is_training, weight_decay=1e-4, name='conv0', padding='SAME')

        conv_layer1_list1 = ConvLayer(
                input_shape=(None, int(self.image_size/4), int(self.image_size/4), 64),
                n_size=3, n_filter=64, stride=1, activation='relu',
                batch_normal=is_training, weight_decay=1e-4, name='conv1_1', padding='SAME')
        conv_layer1_list2 = ConvLayer(
                input_shape=(None, int(self.image_size/4), int(self.image_size/4), 64),
                n_size=3, n_filter=64, stride=1, activation='none',
                batch_normal=is_training, weight_decay=1e-4, name='conv1_2', padding='SAME')


        conv_layer2_list1 = ConvLayer(
                input_shape=(None, int(self.image_size/4)+1, int(self.image_size/4)+1, 64),
                n_size=3, n_filter=128, stride=2, activation='relu',
                #batch_normal=True, weight_decay=1e-4, name='conv2_1', padding='VALID'))
                batch_normal=is_training, weight_decay=1e-4, name='conv2_1', padding='SAME')
        conv_layer2_list2 = ConvLayer(
                input_shape=(None, int(self.image_size/8), int(self.image_size/8), 128),
                n_size=3, n_filter=128, stride=1, activation='none',
                batch_normal=is_training, weight_decay=1e-4, name='conv2_2', padding='SAME')
        conv_layer2_list3 = ConvLayer(
                input_shape=(None, int(self.image_size/4), int(self.image_size/4), 64),
                n_size=1, n_filter=128, stride=2, activation='none',
                #batch_normal=True, weight_decay=1e-4, name='conv2_shortcut', padding='VALID'))
                batch_normal=is_training, weight_decay=1e-4, name='conv2_shortcut', padding='SAME')

        conv_layer3_list1 = ConvLayer(
                input_shape=(None, int(self.image_size/8)+1, int(self.image_size/8)+1, 128),
                n_size=3, n_filter=256, stride=2, activation='relu',
                #batch_normal=True, weight_decay=1e-4, name='conv3_1', padding='VALID'))
                batch_normal=is_training, weight_decay=1e-4, name='conv3_1', padding='SAME')
        conv_layer3_list2 = ConvLayer(
                input_shape=(None, int(self.image_size/16), int(self.image_size/16), 256),
                n_size=3, n_filter=256, stride=1, activation='none',
                batch_normal=is_training, weight_decay=1e-4, name='conv3_2', padding='SAME')
        conv_layer3_list3 = ConvLayer(
                input_shape=(None, int(self.image_size/8), int(self.image_size/8), 128),
                n_size=1, n_filter=256, stride=2, activation='none',
                # batch_normal=True, weight_decay=1e-4, name='conv3_shortcut', padding='VALID'))
                batch_normal=is_training, weight_decay=1e-4, name='conv3_shortcut', padding='SAME')

        conv_layer4_list1 = ConvLayer(
                input_shape=(None, int(self.image_size/16), int(self.image_size/16), 256),
                n_size=3, n_filter=256, stride=1, activation='relu',
                batch_normal=is_training, weight_decay=1e-4, name='conv4_1', padding='SAME')
        conv_layer4_list2 = ConvLayer(
                input_shape=(None, int(self.image_size/16), int(self.image_size/16), 256),
                n_size=3, n_filter=256, stride=1, activation='relu',
                batch_normal=is_training, weight_decay=1e-4, name='conv4_2', padding='SAME')

        dense_layer1 = DenseLayer(
            input_shape=(None, 256),
            hidden_dim=self.n_classes,
            activation='none', dropout=False, keep_prob=None,
            batch_normal=False, weight_decay=1e-4, name='dense1')

        # images_size: 128x128
        # images_pad = tf.pad(images, [[0,0], [3,3], [3,3], [0,0]])
        hidden_conv = conv_layer0_list.get_output(input=images)
        #64xx64
        #print(hidden_conv.get_shape())

        # hidden_pad = tf.pad(hidden_conv, [[0,0], [1,1], [1,1], [0,0]])
        hidden_pool = tf.nn.max_pool(hidden_conv, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
        #32x32
        #print(hidden_pool.get_shape())
        hidden_conv1 = conv_layer1_list1.get_output(input=hidden_pool)
        hidden_conv2 = conv_layer1_list2.get_output(input=hidden_conv1)
        hidden_conv = tf.nn.relu(hidden_pool + hidden_conv2)
        #32x32

        # hidden_pad = tf.pad(hidden_conv, [[0,0], [1,1], [1,1], [0,0]])
        #print(hidden_conv.get_shape())
        hidden_conv1 = conv_layer2_list1.get_output(input=hidden_conv)
        hidden_conv2 = conv_layer2_list2.get_output(input=hidden_conv1)
        hidden_shortcut = conv_layer2_list3.get_output(input=hidden_conv)
        #print("layer2 shortcut")
        #print(hidden_shortcut.get_shape())
        hidden_conv = tf.nn.relu(hidden_shortcut + hidden_conv2)
        #16x16
        #print(hidden_conv.get_shape())
        # hidden_pad = tf.pad(hidden_conv, [[0,0], [1,1], [1,1], [0,0]])
        hidden_conv1 = conv_layer3_list1.get_output(input=hidden_conv)
        hidden_conv2 = conv_layer3_list2.get_output(input=hidden_conv1)
        hidden_shortcut = conv_layer3_list3.get_output(input=hidden_conv)
        hidden_conv = tf.nn.relu(hidden_shortcut + hidden_conv2)
        #8x8
        #print(hidden_conv.get_shape())
        hidden_conv1 = conv_layer4_list1.get_output(input=hidden_conv)
        hidden_conv2 = conv_layer4_list2.get_output(input=hidden_conv1)
        hidden_conv = tf.nn.relu(hidden_conv + hidden_conv2)
        #8x8
        #print("#######################################")
        #print(hidden_conv.get_shape())

        #hidden_pool = tf.nn.avg_pool(hidden_conv, ksize=[1,8,8,1], strides=[1,1,1,1], padding='VALID')
        # global average pooling
        input_dense1 = tf.reduce_mean(hidden_conv, reduction_indices=[1, 2])
        #1x1

        logits = dense_layer1.get_output(input=input_dense1)

        return logits

    def train(self, model_save_dir,  n_epoch=120, batch_size=64):
        #create_record()
        # 构建会话
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 模型保存器


        self.saver = tf.train.Saver(var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, max_to_keep=5)
        # 模型初始化

        if not args.resume:
            self.sess.run(tf.global_variables_initializer())

        #writer = tf.summary.FileWriter('./tensorflow_log/model_adam_weightx10/')
        #outloss = tf.summary.scalar('loss', self.avg_loss)
        #merged = tf.summary.merge([outloss])
        writer = SummaryWriter("./tensorboardX/"+ model_save_dir.split('/')[-1])

        tf.add_to_collection('util_tensor',self.images)
        tf.add_to_collection('util_tensor',self.avg_loss)
        tf.add_to_collection('util_tensor',self.pred)
        tf.add_to_collection('util_tensor',self.accuracy)
        tf.add_to_collection('util_tensor',self.pred)

        # 数据
        filenames=['./data/train_sum80w_win20w_se24w/train{}.tfrecords'.format(i) for i in range(8)]
        filenames_test=['./data/test_new_noshuffle.tfrecords']
        #filenames_test=['./data/train{}.tfrecords'.format(i) for i in range(1)]

        next_element=Tfrecord.dataset(filenames,batch_size=batch_size,epochs=n_epoch)
        next_element_test=Tfrecord.dataset(filenames_test,batch_size=batch_size,epochs=1,istraining=False)
        #next_element=image_data_loader.dataloader(image_dir, batch_size=batch_size, epochs=n_epoch, istraining=False)

        bast_acc = 0.0
        train_interval = 5000

        train_loss = 0.0
        train_correct = 0.0
        train_total = 0
        train_label=[]
        train_pred=[]
        batch_ind = 1
        loop = 1

        with self.sess.as_default():
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess ,coord=coord)

            if args.resume:
                model_path = "/workspace/tensorflow/tensorflow_cls5/model3_xxxxx/model3_3oldmean/model_90000.ckpt"
                self.saver.restore(self.sess, model_path)
                print("\n Restore successful !!!")

            print("\n Building session !!! \n ")
            while True:
                try:
                    #self.BN_training = True
                    #self.logits = self.inference(self.images, is_training=self.BN_training)
                    batch_images, batch_label = self.sess.run([next_element[0],next_element[1]])

                    [_, train_pred_, avg_acc, avg_loss, iteration] = self.sess.run(
                                                        fetches=[self.train_op, self.pred, self.accuracy, self.avg_loss, self.global_step],
                                                        feed_dict={self.images:batch_images,self.labels:batch_label, self.BN_training:True})

                    train_correct += avg_acc * batch_label.shape[0]
                    train_loss += avg_loss# * batch_label.shape[0]
                    train_total += batch_label.shape[0]

                    batch_label=np.argmax(batch_label,axis=1)
                    train_label = np.append(train_label, batch_label)
                    train_pred = np.append(train_pred, train_pred_)

                    progress_bar(batch_ind-1, train_interval, msg=' iteration:{} | Loss: {:.2f} | Acc: {:.2f} ({}/{})   '.format(
                        iteration, train_loss/train_total, 100.*train_correct/train_total, int(train_correct), int(train_total)))
                    batch_ind += 1

                    if iteration % train_interval==0:
                        train_recall = [recall_score(train_label,train_pred,labels=[i],average='micro') for i in range(5)]
                        train_precision = [precision_score(train_label,train_pred,labels=[i],average='micro') for i in range(5)]
                        train_acc = [accuracy_score(train_label,train_pred)]
                        print('train_acc({:.3f}) | listen({:.3f}|{:.3f}) write({:.3f}|{:.3f}) handup({:.3f}|{:.3f}) positive({:.3f}|{:.3f}) negative({:.3f}|{:.3f})\n'.format(
                            train_acc[0], train_recall[0],train_precision[0], train_recall[1],train_precision[1], train_recall[2],train_precision[2],
                            train_recall[3],train_precision[3], train_recall[4],train_precision[4]))
                        log.write('train_acc({:.3f}) | listen({:.3f}|{:.3f}) write({:.3f}|{:.3f}) handup({:.3f}|{:.3f}) positive({:.3f}|{:.3f}) negative({:.3f}|{:.3f})\n'.format(
                            train_acc[0], train_recall[0],train_precision[0], train_recall[1],train_precision[1], train_recall[2],train_precision[2],
                            train_recall[3],train_precision[3], train_recall[4],train_precision[4]))

                        train_loss = 0.0
                        train_correct = 0.0
                        train_total = 0
                        train_label=[]
                        train_pred=[]
                        batch_ind = 1
                        #print('set 00000000000')
                        valid_loss = 0.0
                        valid_correct = 0.0
                        val_total = 0
                        batch_index = 1
                        test_label=[]
                        test_pred=[]

                        print(">>> Test: %d <<<<" % (loop))
                        loop += 1

                        #self.BN_training = False
                        #self.logits = self.inference(self.images, is_training=self.BN_training)
                        while True:
                            try:
                                batch_images,batch_labels=self.sess.run([next_element_test[0],next_element_test[1]])
                                [pred, avg_acc, avg_loss] =self.sess.run(fetches=[self.pred, self.accuracy, self.avg_loss],
                                                                    feed_dict={self.images:batch_images,self.labels:batch_labels, self.BN_training:False})

                                valid_correct += avg_acc * batch_labels.shape[0]
                                valid_loss += avg_loss #* batch_labels.shape[0]
                                val_total += batch_labels.shape[0]

                                batch_labels=np.argmax(batch_labels,axis=1)
                                test_label = np.append(test_label, batch_labels)
                                test_pred = np.append(test_pred, pred)

                                    #274
                                progress_bar(batch_index, 274, msg=' iteration: {} | Loss: {:.2f} | Acc: {:.2f} ({}/{})  '.format(
                                    iteration, valid_loss/val_total, 100.*valid_correct/val_total, int(valid_correct), int(val_total)))
                                batch_index += 1
                                #writer.add_summary(summary, global_step=iteration)
                            except tf.errors.OutOfRangeError:
                                #cls_dict = {'listen':0, 'write':1, 'handup':2, 'positive':3, 'negative':4}  # wenqaing
                                test_recall = [recall_score(test_label,test_pred,labels=[i],average='micro') for i in range(5)]
                                test_precision = [precision_score(test_label,test_pred,labels=[i],average='micro') for i in range(5)]
                                test_acc = [accuracy_score(test_label,test_pred)]
                                print('test_acc:({:.4f}) | listen({:.3f}|{:.3f}) write({:.3f}|{:.3f}) handup({:.3f}|{:.3f}) positive({:.3f}|{:.3f}) negative({:.3f}|{:.3f})\n'.format(
                                    test_acc[0], test_recall[0],test_precision[0], test_recall[1],test_precision[1], test_recall[2],test_precision[2],
                                              test_recall[3],test_precision[3], test_recall[4],test_precision[4]))

                                log.write('test_acc:({:.4f}) | listen({:.3f}|{:.3f}) write({:.3f}|{:.3f}) handup({:.3f}|{:.3f}) positive({:.3f}|{:.3f}) negative({:.3f}|{:.3f})\n\n'.format(
                                        test_acc[0], test_recall[0],test_precision[0], test_recall[1],test_precision[1], test_recall[2],test_precision[2],
                                        test_recall[3],test_precision[3], test_recall[4],test_precision[4]))

                                next_element_test=Tfrecord.dataset(filenames_test,batch_size=batch_size,epochs=1,istraining=False)

                                writer.add_scalars('data/Acc_group', {'train_acc': train_acc[0], 'test_acc': test_acc[0]}, iteration)
                                if valid_correct > bast_acc:
                                    bast_acc = valid_correct
                                    self.saver.save(self.sess, os.path.join(model_save_dir, "model_%d.ckpt"%(iteration)))
                                    print("#############  saving...  ##########\n")
                                    log.write("#############  saving...  ##########\n\n")
                                break

                except tf.errors.OutOfRangeError:
                    print("End of the Train")
                    break
            coord.request_stop()
            coord.join(threads)
            writer.close()
        self.sess.close()


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--resume', '-r', action='store_true', help='run continue ')
    parser.add_argument('--recall_dir', dest='recall_dir', help='recall_dir',
                        default='./demo/recall_dir', type=str)

    args = parser.parse_args()
    ###################################################################################################################
    model_save_dir = "/workspace/tensorflow/tensorflow_cls5/model3_xxxxx/model3_oldnorm"
    log = open('./logs/{}.txt'.format(model_save_dir.split('/')[-1]), 'w')
    GPU_index = '0'
    print("\n############################# save in << %s >>  and on << %s >>GPU ##############################\n"%(model_save_dir, GPU_index ))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index

    convnet = ConvNet(n_channel=3, n_classes=5, image_size=128, n_layers=5)
    convnet.train(model_save_dir, n_epoch=120, batch_size=64)
