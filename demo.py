# -*- encoding: utf8 -*-
# author: ronniecao
import sys
import os

import tensorflow as tf

from layer.basicblock import ConvLayer
from layer.dense_layer import DenseLayer


from utils.utils import progress_bar
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, accuracy_score
import numpy as np
from PIL import Image
import cv2
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
        self.labels = tf.placeholder(dtype=tf.int64, shape=[None], name='labels')  ##############
        self.BN_training = tf.placeholder(dtype=tf.bool, name='BN_mode')
        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step')

        # 网络输出
        self.logits = self.inference(self.images,istraining=self.BN_training)
        self.soft_out = tf.nn.softmax(self.logits, axis=-1)
        self.objective = tf.reduce_sum(tf.losses.sparse_softmax_cross_entropy(logits=self.logits, labels=self.labels))

        tf.add_to_collection('losses', self.objective)
        self.avg_loss = tf.add_n(tf.get_collection('losses'))

        # 观察值
        self.pred = tf.argmax(self.logits, 1)
        correct_prediction = tf.equal(self.labels, self.pred)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


    def inference(self, images,istraining = False):


        conv_layer0_list = ConvLayer(
                input_shape=(None, self.image_size+6, self.image_size+6, self.n_channel),
                n_size=7, n_filter=64, stride=2, activation='relu',
                batch_normal=istraining, weight_decay=1e-4, name='conv0', padding='SAME')

        conv_layer1_list1 = ConvLayer(
                input_shape=(None, int(self.image_size/4), int(self.image_size/4), 64),
                n_size=3, n_filter=64, stride=1, activation='relu',
                batch_normal=istraining, weight_decay=1e-4, name='conv1_1', padding='SAME')
        conv_layer1_list2 = ConvLayer(
                input_shape=(None, int(self.image_size/4), int(self.image_size/4), 64),
                n_size=3, n_filter=64, stride=1, activation='none',
                batch_normal=istraining, weight_decay=1e-4, name='conv1_2', padding='SAME')

        conv_layer2_list1 = ConvLayer(
                input_shape=(None, int(self.image_size/4)+1, int(self.image_size/4)+1, 64),
                n_size=3, n_filter=128, stride=2, activation='relu',
                batch_normal=istraining, weight_decay=1e-4, name='conv2_1', padding='SAME')
        conv_layer2_list2 = ConvLayer(
                input_shape=(None, int(self.image_size/8), int(self.image_size/8), 128),
                n_size=3, n_filter=128, stride=1, activation='none',
                batch_normal=istraining, weight_decay=1e-4, name='conv2_2', padding='SAME')
        conv_layer2_list3 = ConvLayer(
                input_shape=(None, int(self.image_size/4), int(self.image_size/4), 64),
                n_size=1, n_filter=128, stride=2, activation='none',
                batch_normal=istraining, weight_decay=1e-4, name='conv2_shortcut', padding='SAME')

        conv_layer3_list1 = ConvLayer(
                input_shape=(None, int(self.image_size/8)+1, int(self.image_size/8)+1, 128),
                n_size=3, n_filter=256, stride=2, activation='relu',
                batch_normal=istraining, weight_decay=1e-4, name='conv3_1', padding='SAME')
        conv_layer3_list2 = ConvLayer(
                input_shape=(None, int(self.image_size/16), int(self.image_size/16), 256),
                n_size=3, n_filter=256, stride=1, activation='none',
                batch_normal=istraining, weight_decay=1e-4, name='conv3_2', padding='SAME')
        conv_layer3_list3 = ConvLayer(
                input_shape=(None, int(self.image_size/8), int(self.image_size/8), 128),
                n_size=1, n_filter=256, stride=2, activation='none',
                batch_normal=istraining, weight_decay=1e-4, name='conv3_shortcut', padding='SAME')

        conv_layer4_list1 = ConvLayer(
                input_shape=(None, int(self.image_size/16), int(self.image_size/16), 256),
                n_size=3, n_filter=256, stride=1, activation='relu',
                batch_normal=istraining, weight_decay=1e-4, name='conv4_1', padding='SAME')
        conv_layer4_list2 = ConvLayer(
                input_shape=(None, int(self.image_size/16), int(self.image_size/16), 256),
                n_size=3, n_filter=256, stride=1, activation='relu',
                batch_normal=istraining, weight_decay=1e-4, name='conv4_2', padding='SAME')

        dense_layer1 = DenseLayer(
                input_shape=(None, 256),
                hidden_dim=self.n_classes,
                activation='none', dropout=False, keep_prob=None,
                batch_normal=False, weight_decay=1e-4, name='dense1')

        # images_size: 128x128
        hidden_conv = conv_layer0_list.get_output(input=images)
        #64xx64

        hidden_pool = tf.nn.max_pool(hidden_conv, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
        #32x32

        hidden_conv1 = conv_layer1_list1.get_output(input=hidden_pool)
        hidden_conv2 = conv_layer1_list2.get_output(input=hidden_conv1)
        hidden_conv = tf.nn.relu(hidden_pool + hidden_conv2)
        #32x32

        hidden_conv1 = conv_layer2_list1.get_output(input=hidden_conv)
        hidden_conv2 = conv_layer2_list2.get_output(input=hidden_conv1)
        hidden_shortcut = conv_layer2_list3.get_output(input=hidden_conv)
        hidden_conv = tf.nn.relu(hidden_shortcut + hidden_conv2)
        #16x16

        hidden_conv1 = conv_layer3_list1.get_output(input=hidden_conv)
        hidden_conv2 = conv_layer3_list2.get_output(input=hidden_conv1)
        hidden_shortcut = conv_layer3_list3.get_output(input=hidden_conv)
        hidden_conv = tf.nn.relu(hidden_shortcut + hidden_conv2)
        #8x8

        hidden_conv1 = conv_layer4_list1.get_output(input=hidden_conv)
        hidden_conv2 = conv_layer4_list2.get_output(input=hidden_conv1)
        hidden_conv = tf.nn.relu(hidden_conv + hidden_conv2)
        #8x8

        #hidden_pool = tf.nn.avg_pool(hidden_conv, ksize=[1,8,8,1], strides=[1,1,1,1], padding='VALID')
        #global average pooling
        input_dense1 = tf.reduce_mean(hidden_conv, reduction_indices=[1, 2])
        #1x1

        logits = dense_layer1.get_output(input=input_dense1)

        return logits

    def badcase_print(self, mode, result_dir, soft_out, batch_preds, batch_labels, batch_images, batch_index):
        cls_dict = {'listen':0, 'write':1, 'handup':2, 'positive':3, 'negative':4 }
        label_dict = {0:'listen', 1:'write', 2:'handup', 3:'positive', 4:'negative'}   #weniqiang

        for label in cls_dict.keys():
            recall_path = os.path.join(result_dir, label)
            if not os.path.exists(recall_path):
                os.makedirs(recall_path)

        topk = 3
        topk_index = np.argsort(-soft_out, axis=-1)  #sort in decline
        topk_prob = -np.sort(-soft_out)

        FN_index = np.nonzero(batch_preds != batch_labels)
        img_num = 0
        if len(FN_index[0]) == 0:
            return
        global num_badcase
        num_badcase += len(FN_index[0])
        for index in FN_index[0]:

            batch_images = np.asarray(batch_images)
            #与数据预处理过程互逆
            image = batch_images[index].reshape(128,128,3)

            #image = image + 0.5
            image = self.de_image_normalize(image)

            img = np.clip((image * 255).astype(np.int16), 0, 255)

            r,g,b=cv2.split(img)
            img = cv2.merge([b,g,r])
            caption = np.zeros((40+20*topk, img.shape[1], 3))
            img_pad  = np.array(np.vstack((caption, img)),dtype = np.uint8)
            cv2.putText(img_pad, 'label:'+ label_dict[batch_labels[index]] , (1,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(img_pad, 'pred:'+ label_dict[batch_preds[index]] , (1,25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            for i in range(topk):
                cv2.putText(img_pad, '{}: {:.3f}'.format(label_dict[topk_index[index,i]], topk_prob[index,i]) , (1,45+20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if mode == "recall":
                img_path = os.path.join(os.path.join(result_dir, label_dict[batch_labels[index]]), '%s_%d_%d.jpg'%(label_dict[batch_labels[index]] ,batch_index, img_num))
            elif mode == "precision":
                img_path = os.path.join(os.path.join(result_dir, label_dict[batch_preds[index]]), '%s_%d_%d.jpg'%(label_dict[batch_labels[index]] ,batch_index, img_num))
            else:
                return
            cv2.imwrite(img_path, img_pad)
            img_num += 1

    def de_image_normalize(sellf, image):
        # image de_normalize
        #mean = np.array([0.4679, 0.4169, 0.3915])  #new
        #std = np.array([0.3115, 0.3080, 0.3191])
        mean=np.array([0.4111, 0.3296, 0.2866])    #old
        #std=np.array([0.2845, 0.2700, 0.2755])
        #image = image * std
        image = image + mean
        return image

    def image_normalize(self, image):
        # image normalize
        #mean = np.array([0.4679, 0.4169, 0.3915])  #new
        #std = np.array([0.3115, 0.3080, 0.3191])
        mean=np.array([0.4111, 0.3296, 0.2866])    #old
        #std=np.array([0.2845, 0.2700, 0.2755])
        image = (image - mean)# / std
        return image

    def evalute(self, n_epoch=1, batch_size=64):

        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.saver = tf.train.Saver(var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, max_to_keep=3)

        # 数据
        #filenames_test=['./data/test_new_noshuffle.tfrecords']
        #next_element_test=Tfrecord.dataset(filenames_test,batch_size=batch_size,epochs=1,istraining=False)

        print("\n Building session !!! \n ")
        with self.sess.as_default():
            print(" Restore model ......\n")
            self.saver.restore(self.sess, model_path)
            print(" Restore successful !!! \n ")

            valid_loss = 0.0
            valid_correct = 0.0
            val_total = 0
            batch_index = 1
            test_label=[]
            test_pred=[]

            print("\n>>> Test:<<<<\n")

            for class_dir in os.listdir(image_test_dir):
                class_path = os.path.join(image_test_dir, class_dir)
                assert os.path.isdir(class_path)

                image_name_list = os.listdir(class_path)
                image_path_label = [[os.path.join(class_path, item), cls_dict[class_dir]] for item in image_name_list]

                batch_image = []
                batch_label = []
                for index, image_info in enumerate(image_path_label):
                    img_path = image_info[0]
                    label = image_info[1]

                    image = Image.open(os.path.abspath(img_path)).convert('RGB')
                    image = image.resize((128,128))

                    image = np.asarray(image)
                    image = image.astype(np.float32)*(1/255)# - 0.5
                    image = self.image_normalize(image)

                    label = np.array(label, dtype=np.int64)
                    batch_image.append(image)
                    batch_label.append(label)

                    if (index+1)%batch_size !=0 and (index+1) !=len(image_path_label):
                        continue
                    else:
                        batch_images = np.stack(batch_image, axis=0)
                        batch_labels = np.stack(batch_label, axis=0)
                        batch_image = []
                        batch_label = []

                        [logits, soft_out, pred, avg_acc, avg_loss] =self.sess.run(fetches=[self.logits ,self.soft_out,self.pred, self.accuracy, self.avg_loss],
                                                feed_dict={self.images:batch_images,self.labels:batch_labels,self.BN_training:False})

                        #used for print badcase
                        if args.print_badcase:
                            self.badcase_print("precision", badcase_dir, soft_out, pred, batch_labels, batch_images, batch_index)

                        valid_correct += avg_acc * batch_labels.shape[0]
                        valid_loss += avg_loss
                        val_total += batch_labels.shape[0]

                        #batch_labels=np.argmax(batch_labels,axis=1)
                        test_label = np.append(test_label, batch_labels)
                        test_pred = np.append(test_pred, pred)


                        progress_bar(batch_index, 274*100, msg=' iteration: {} | Loss: {:.2f} | Acc: {:.2f} ({}/{})  '.format(
                            215000, valid_loss/val_total, 100.*valid_correct/val_total, int(valid_correct), int(val_total)))
                        batch_index += 1

            test_recall = [recall_score(test_label,test_pred,labels=[i],average='micro') for i in range(5)]
            test_precision = [precision_score(test_label,test_pred,labels=[i],average='micro') for i in range(5)]
            test_acc = [accuracy_score(test_label,test_pred)]
            print('\ntest_acc:({:.4f}) | listen({:.3f}|{:.3f}) write({:.3f}|{:.3f}) handup({:.3f}|{:.3f}) positive({:.3f}|{:.3f}) negative({:.3f}|{:.3f})\n'.format(
                test_acc[0], test_recall[0],test_precision[0], test_recall[1],test_precision[1], test_recall[2],test_precision[2],
                          test_recall[3],test_precision[3], test_recall[4],test_precision[4]))
            print("\nTotal badcase num is: >>>>>  %d  <<<<<\n"%(num_badcase))

        self.sess.close()


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Tensorflow 5cls Training')
    parser.add_argument('--print_badcase', '-p', action='store_true', help='used for print badcase...')
    args = parser.parse_args()


    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    cls_dict = {'listen':0, 'write':1, 'handup':2, 'positive':3, 'negative':4}  # wenqaing
    num_badcase = 0

    model_path = "./model/model3_badcase_relabel_finetune/model_1255000.ckpt"
    image_test_dir = r'/workspace/xpy/data/new_standard/test_new'
    badcase_dir = r"./badcase/model3_badcase"

    convnet = ConvNet(n_channel=3, n_classes=5, image_size=128)
    convnet.evalute(n_epoch=1, batch_size=64)
