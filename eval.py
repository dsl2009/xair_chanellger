#coding=utf-8
from threading import Thread
import socket
import tensorflow as tf
from nets import inception
import shutil
import os
import cv2
import numpy as np
import json
from tensorflow.contrib import slim
from nets import inception_v3
import glob
#log_dir = '/home/dsl/all_check/log_nasa_sgd20'
checkpoints_dir = '/home/dsl/all_check/pig/pig_nasa_dsl_final'
labels_file = 'labels.json'
labels_to_name =  json.loads(open(labels_file).read())
print(labels_to_name)
from skimage import io

def native_build(image_size, path):

    iv1 = io.imread(path)
    iv1 = iv1[:,:,0:3]
    bc = np.asarray(iv1, dtype=np.float32) / 255

    w, h, c = bc.shape
    ot = []
    crop_size =min([w,h])
    if crop_size == w:
        bbox_w_start =0

    for s in range(5):
        if crop_size == w:
            bbox_w_start = 0
        else:
            bbox_w_start = np.random.randint(0, w - crop_size)
        if crop_size == h:
            bbox_h_start = 0
        else:
            bbox_h_start = np.random.randint(0, h - crop_size)


        bbox_h_size = crop_size
        bbox_w_size = crop_size

        img = bc[bbox_w_start:bbox_w_start + bbox_w_size, bbox_h_start:bbox_h_size + bbox_h_start]
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        image = img - 0.5
        image = image * 2.0
        ot.append(image)

    image = np.reshape(np.asarray(ot), (5, image_size, image_size, 3))
    return image


def recongnizev3():
    image_size = 299
    dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/pred_step1'
    with tf.Graph().as_default():
        #with tf.device('/cpu:0'):
        processed_images = tf.placeholder(shape=(5,299,299,3),dtype=tf.float32)
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, end_points = inception_v3.inception_v3(processed_images, num_classes=10, dropout_keep_prob=1.0, is_training=False)
        logits = tf.nn.softmax(logits,axis=1)
        saver = tf.train.Saver()
        total = 0
        right = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, '/home/dsl/all_check/aichallenger/ai_chanellger/trained/model.ckpt-92969')
            l1 = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/ai_challenger_pdr2018_testA_20180905/AgriculturalDisease_testA/images/*.*')
            #l1 = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/valid/AgriculturalDisease_validationset/images/*.*')


            for s in l1:
                pi = native_build(image_size, s)
                np_image, pp = sess.run([processed_images, logits],feed_dict={processed_images:pi})
                pp = np.sum(pp,axis=0)/5
                sorted_inds = [i[0] for i in sorted(enumerate(-pp), key=lambda x: x[1])]
                lbsnn = s.replace('\\','/').split('/')[-2]
                name = labels_to_name[str(sorted_inds[0])]
                step_dr = os.path.join(dr,name)
                if not os.path.exists(step_dr):

                    os.makedirs(step_dr)
                shutil.copy(s,step_dr)


recongnizev3()