import os
import cv2
import numpy as np
import matplotlib.pylab as plt
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import FLAGS
from nets.ssd_vgg_512_Multi import SSDNet_vgg_512 as SSDNet
import utils.utils as utils
import dataset.dataSet_utils as data_utils

train_result_path = r'img_result\train_result_Tail'
test_result_path = r'img_result\test_result_Tail'

log_step = 100
if not os.path.exists(train_result_path):
    os.makedirs(train_result_path)
if not os.path.exists(test_result_path):
    os.makedirs(test_result_path)

def draw_boxes(img,localizations,scores,labels,img_name,threshold=0.5):
    index = np.nonzero(np.logical_and(scores > threshold, labels > 0))
    localizations = localizations * 512
    img = img * 255
    img = np.uint8(img)
    cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
    localizations = np.int32(localizations)
    index=np.array(index)
    index=np.reshape(index,[-1])
    for i in range(len(index)):
        idx=index[i]
        loc = localizations[idx,:]
        red_color=int(scores[idx]*255)
        cv2.rectangle(img, (loc[1], loc[0]), (loc[3], loc[2]), (0, 0, red_color),1)
        cv2.putText(img,str(labels[idx])+'(%.2f)'%scores[idx],(loc[1]+20,loc[0]+10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
    cv2.imwrite(img_name, img)
def drar_hp(img,feat_map):
    img = img * 255
    img = np.uint8(img)
    # cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
    hp_map=np.zeros((512,512))
    for i in range(5):
        temp=feat_map[i]
        temp[temp>0]=1
        temp = temp * 255
        temp = np.uint8(temp)
        hp_map+=cv2.resize(temp,(512,512),interpolation=cv2.INTER_AREA)
    cv2.normalize(hp_map,hp_map,0.,1.,cv2.NORM_MINMAX)
    hp_map=hp_map*255
    hp_map=np.uint8(hp_map)
    plt.subplot(211)
    plt.imshow(img)
    plt.subplot(212)
    plt.imshow(hp_map)
    plt.pause(0.01)
def load_finetuning(sess):
    variables_to_restore = utils.get_variables_to_restore(['vgg_16'], ['Adam', 'Adam_1', 'vgg_16/fc8/weights:0',
                                                                           'vgg_16/fc8/biases:0'])
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, os.path.join(FLAGS.model_path, 'vgg_16.ckpt'))

ssd_nets=SSDNet()
#读取数据
with tf.name_scope('train'):
    train_dataSet = data_utils.get_dataSet(FLAGS.dataSet)
    """
      DatasetDataProviders从数据集中提供数据. 通过配置，可以同时使用多个readers或者使用单个reader提供数据。此外，被读取的数据
    可以被打乱顺序。比如，使用一个单线程读取数据而不打乱顺序的例子如下：
        pascal_voc_data_provider = DatasetDataProvider(
          slim.datasets.pascal_voc.get_split('train'),shuffle=False)
        images, labels = pascal_voc_data_provider.get(['images', 'labels'])
      使用多个readers同时读取数据并且打乱顺序的例子如下：
        pascal_voc_data_provider = DatasetDataProvider(
          slim.datasets.pascal_voc.Dataset(),num_readers=10, shuffle=True)
        images, labels = pascal_voc_data_provider.get(['images', 'labels'])
      同样地，我们可以分开请求相同样本的不同属性，比如：
        [images] = pascal_voc_data_provider.get(['images'])
        [labels] = pascal_voc_data_provider.get(['labels'])
        参数:
      dataset: 一个dataset类的实例
      num_readers: 使用的并行读取器的数量
      reader_kwargs: reader的一个可选的字典参数
      shuffle:读取时是否打乱顺序
      num_epochs: 每个数据源被读取的次数，如果为None，这个数据集将会被无限次读取。
      common_queue_capacity: 公共队列的容量。
      common_queue_min: 公共队列出队后的最小元素个数
      record_key: 记录关键字
      seed: 用于打乱顺序的seed
      scope: 可选的该操作scope名
    """
    train_provider = slim.dataset_data_provider.DatasetDataProvider(dataset=train_dataSet,
                                                                  num_readers=3,
                                                                  common_queue_capacity=FLAGS.batch_size*5,
                                                                  common_queue_min=FLAGS.batch_size*2, shuffle=True)
    [train_img, train_bboxes, train_labels] = train_provider.get(['img','bndboxes','labels'])
    [train_img, train_bboxes, train_labels] = data_utils.preprocess(train_img, train_bboxes, train_labels)

    [train_img, train_bboxes, train_labels] = data_utils.data_argument(train_img, train_bboxes, train_labels)

    ssd_anchors = ssd_nets.place_anchors()

    glocalizations, gscores, glabels, bbboxes = ssd_nets.encode_bboxes(ssd_anchors, train_bboxes, train_labels)
    r = tf.train.shuffle_batch(utils.reshape_list([train_img, glocalizations, gscores,glabels]),
                     batch_size=FLAGS.batch_size,
                     num_threads=3,
                     capacity=3*FLAGS.batch_size, min_after_dequeue=FLAGS.batch_size)

    batch_shape = [1]+[len(ssd_anchors)]*3

    b_img, b_localizations, b_scores, b_labels = utils.reshape_list(r, shape=batch_shape)

    batch_queue = slim.prefetch_queue.prefetch_queue(utils.reshape_list([b_img,b_localizations,b_scores,b_labels]),
                                                   capacity=3*FLAGS.batch_size,num_threads=3)

    train_img, train_bboxes, train_scores, train_labels = utils.reshape_list(batch_queue.dequeue(),
                                                                         batch_shape)

    train_op = ssd_nets.train_op(train_img, train_bboxes, train_scores, train_labels, bbboxes)
    train_test_op = ssd_nets.test_op(train_img, ssd_anchors, reuse=True)
with tf.name_scope('test'):
    test_dataSet = data_utils.get_dataSet(FLAGS.testDataSet)
    test_provider = slim.dataset_data_provider.DatasetDataProvider(dataset=test_dataSet,
                                                                   num_readers=3,
                                                                   common_queue_capacity=FLAGS.batch_size * 5,
                                                                   common_queue_min=FLAGS.batch_size * 2,
                                                                   shuffle=False)
    [test_img, test_bboxes, test_labels] = test_provider.get(['img', 'bndboxes', 'labels'])
    [test_img, test_bboxes, test_labels] = data_utils.preprocess(test_img, test_bboxes, test_labels)
    ssd_anchors = ssd_nets.place_anchors()
    glocalizations, gscores, glabels,bbboxes  = ssd_nets.encode_bboxes(ssd_anchors, test_bboxes, test_labels)
    r = tf.train.batch(utils.reshape_list([test_img, glocalizations, gscores, glabels]),
                       batch_size=1,
                       num_threads=3,
                       capacity=3 * FLAGS.batch_size)
    batch_shape = [1] + [len(ssd_anchors)] * 3
    b_img, b_localizations, b_scores, b_labels = utils.reshape_list(r, shape=batch_shape)
    batch_queue = slim.prefetch_queue.prefetch_queue(
        utils.reshape_list([b_img, b_localizations, b_scores, b_labels]),
        capacity=3 * FLAGS.batch_size, num_threads=3)
    test_img, test_bboxes, test_scores, test_labels = utils.reshape_list(batch_queue.dequeue(),
                                                                         batch_shape)
    validate_op = ssd_nets.validate_op(test_img, test_bboxes, test_scores,
                                   test_labels, ssd_anchors, reuse=True)

summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    if not os.path.exists(FLAGS.model_path):
        os.makedirs(FLAGS.model_path)
    if 'checkpoint' in os.listdir(FLAGS.model_path):
        print('restore from last model....')
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('restore ' + ckpt.model_checkpoint_path)
    else:
        print('restore partial model vgg16')
        variables_to_restore = utils.get_variables_to_restore(['vgg_16'], ['Adam', 'Adam_1'])
        saver = tf.train.Saver(variables_to_restore)

        saver.restore(sess, os.path.join('model', 'vgg_16.ckpt'))
    saver = tf.train.Saver(max_to_keep=None)
    merge_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
    try:
        while not coord.should_stop():
            begin = time.time()
            [_, train_loss, global_step, num, glocalizations] = sess.run(train_op)

            if np.mod(global_step, log_step) == 0:

                [img, localization, scores, labels, index, logits] = sess.run(train_test_op)
                draw_boxes(img, localization[index, :], scores[index], labels[index],
                           os.path.join(train_result_path, '%d.jpg' % global_step), threshold=0.5)

                [img, localization, scores, labels, index, _,validate_loss] = sess.run(validate_op)
                draw_boxes(img, localization[index, :], scores[index], labels[index],
                           os.path.join(test_result_path, '%d.jpg' % global_step), threshold=0.5)
                end = time.time()
                seccond = end - begin
                fps = log_step * FLAGS.batch_size / seccond

                summary = sess.run(merge_op)
                summary_writer.add_summary(summary, global_step)

                print("step:%d,positive num:%s,train loss:%f,validate_loss:%f,%f fps"
                      % (global_step, num, train_loss, validate_loss, fps))
            if np.mod(global_step, 500) == 1:
                saver.save(sess, os.path.join(FLAGS.model_path, 'model'), global_step)
    except tf.errors.OutOfRangeError:
        print('training completed')
    finally:
        coord.request_stop()
    coord.join(threads=threads)
if __name__=='__main__':
    tf.app.run()


