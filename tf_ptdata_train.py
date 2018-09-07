import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from preprocessing import inception_preprocessing
from nets import inception_v3
import os
import time
import json
import numpy as np
from tensorflow.contrib import slim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
dataset_dir = ''
log_dir = 'log'
checkpoint_file = '/home/dsl/all_check/inception_v3.ckpt'
num_epochs = 40
batch_size = 8
initial_learning_rate = 0.01
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=(480,320)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=(480,320)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}
imagenet_data = ImageFolder('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/org/train'
                            '/AgriculturalDisease_trainingset/柑桔黄龙病',
                                 transform=data_transforms['val'])
data_loader = DataLoader(imagenet_data, batch_size=8, shuffle=True,drop_last=True)

def run():
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tf.logging.set_verbosity(tf.logging.INFO)
    images = tf.placeholder(shape=(batch_size,480, 320, 3) ,dtype=tf.float32)
    labels = tf.placeholder(shape=[batch_size] ,dtype=tf.int32)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, end_points = inception_v3.inception_v3(images, num_classes=2, is_training=True, spatial_squeeze=False,
                                                       global_pool=True)
    exclude = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
    one_hot_labels = slim.one_hot_encoding(labels, 2)
    logits = tf.squeeze(logits, axis=[1, 2])
    loss = slim.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
    tf.losses.add_loss(loss)
    total_loss = tf.losses.get_total_loss()
    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.exponential_decay(
        learning_rate=initial_learning_rate,
        global_step=global_step,
        decay_steps=2000,
        decay_rate=learning_rate_decay_factor,
        staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    train_op = slim.learning.create_train_op(total_loss, optimizer)
    predictions = tf.argmax(slim.softmax(logits), 1)
    probabilities = slim.softmax(logits)
    accuracy, accuracy_update = tf.metrics.accuracy( labels,predictions)
    metrics_op = tf.group(accuracy_update, probabilities)
    tf.summary.scalar('losses/Total_Loss', total_loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('learning_rate', lr)
    my_summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(variables_to_restore)
    def restore_fn(sess):
        return saver.restore(sess, checkpoint_file)
    sv = tf.train.Supervisor(logdir=log_dir, summary_op=None, init_fn=restore_fn)
    with sv.managed_session() as sess:
        for step in range(num_epochs):
            for ix, (data, label) in enumerate(data_loader):
                dt = data.numpy()
                lb = label.numpy()
                dt = np.transpose(dt, [0, 2, 3, 1])
                fd = {images: dt, labels: lb}
                total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op], feed_dict=fd)
                if ix%10==0:
                    print(global_step_count,total_loss)

                if ix % 100 == 0:
                    learning_rate_value, accuracy_value = sess.run([lr, accuracy], feed_dict=fd)
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    logging.info('Current Streaming Accuracy: %s', accuracy_value)

                    logits_value, probabilities_value, predictions_value, labels_value = sess.run(
                        [logits, probabilities, predictions, labels], feed_dict=fd)




if __name__ == '__main__':
    run()