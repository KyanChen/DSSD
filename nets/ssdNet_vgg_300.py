import math
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

import detection.ssd.utils.utils as utils


class SSDNet(object):
    def __init__(self):
        self.anchors_blocks = ('block4', 'block7', 'block8', 'block9', 'block10', 'block11')  # block的索引
        self.anchors_steps = (8, 16, 32, 64, 100, 300)
        self.offset = 0.5
        self.anchors_ratios = [[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]]
        self.anchors_sizes = [(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)]
        self.feat_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        self.img_size = 300
        self.num_classes = 21
        self.losses = 0

    def place_anchors(self):
        """
        设置anchors
        :return: 
        """
        ssd_anchors = []
        for i, sizes in enumerate(self.anchors_sizes):
            anchors = self.place_one_anchors(self.img_size, self.feat_shapes[i], self.anchors_ratios[i]
                                             , sizes, self.anchors_steps[i], self.offset)
            ssd_anchors.append(anchors)
        return ssd_anchors

    def place_one_anchors(self, img_size, shape, ratios, sizes, step, offset):
        """

        :param img_size: 图像的大小
        :param shape: 
        :param ratios: 
        :param sizes: 
        :param step: 
        :param offset: 
        :return: 返回head的锚点以及候选框的大小
        """
        y, x = np.mgrid[0:shape[0], 0:shape[1]]
        y = (y.astype(np.float32) + offset) * step / img_size
        x = (x.astype(np.float32) + offset) * step / img_size

        y = tf.expand_dims(y, axis=-1)
        x = tf.expand_dims(x, axis=-1)

        anchors_len = len(ratios)+2
        h = np.zeros((anchors_len,), np.float32)
        w = np.zeros((anchors_len,), np.float32)
        h[0] = sizes[0] / img_size
        w[0] = sizes[0] / img_size
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_size
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_size
        for i, ratio in enumerate(ratios):
            h[i+2] = sizes[0] / img_size / math.sqrt(ratio)
            w[i+2] = sizes[0] / img_size * math.sqrt(ratio)
        return y, x, h, w

    def encode_bboxes(self, ssd_anchors, bboxes, labels):
        feat_locs = []
        feat_scores = []
        feat_labels = []
        for i, anchors in enumerate(ssd_anchors):
            loc, score, label = self.encode_one_bboxes(anchors, bboxes, labels)
            feat_locs.append(loc)
            feat_scores.append(score)
            feat_labels.append(label)
        return feat_locs, feat_scores, feat_labels

    def encode_one_bboxes(self, anchors, bboxes, labels, threshold=0.5):
        yref, xref, href, wref = anchors
        ymin = yref - href / 2
        xmin = xref - wref / 2
        ymax = yref + href / 2
        xmax = xref + wref / 2
        anchors_area = (ymax - ymin) * (xmax - xmin)

        anchor_shape = np.shape(ymin)
        feat_ymin = tf.zeros(anchor_shape, tf.float32)
        feat_xmin = tf.zeros(anchor_shape, tf.float32)
        feat_ymax = tf.ones(anchor_shape, tf.float32)
        feat_xmax = tf.ones(anchor_shape, tf.float32)
        feat_score = tf.zeros(anchor_shape, tf.float32)
        feat_labels = tf.zeros(anchor_shape, tf.int64)

        def jaccard_similarity(bbox):
            """
            计算anchors与任意bbox的jaccard相似度
            :param bbox: 
            :return: 返回交并比
            """
            inter_ymin = tf.maximum(ymin, bbox[0])
            inter_xmin = tf.maximum(xmin, bbox[1])
            inter_ymax = tf.minimum(ymax, bbox[2])
            inter_xmax = tf.minimum(xmax, bbox[3])

            inter_h = tf.maximum(inter_ymax - inter_ymin, 0.)
            inter_w = tf.maximum(inter_xmax - inter_xmin, 0.)
            inter_area = inter_h * inter_w
            total_area = anchors_area - inter_area + (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
            return tf.div(inter_area, total_area)

        def body(i, feat_ymin, feat_xmin, feat_ymax, feat_xmax, feat_score, feat_labels):
            bbox = bboxes[i]

            jaccard = jaccard_similarity(bbox)
            mask = tf.greater(jaccard, feat_score)
            mask = tf.logical_and(mask, tf.greater_equal(jaccard, threshold))
            imask = tf.cast(mask, tf.int64)
            fmask = tf.cast(mask, tf.float32)
            feat_ymin = tf.where(mask, fmask * bbox[0], feat_ymin)
            feat_xmin = tf.where(mask, fmask * bbox[1], feat_xmin)
            feat_ymax = tf.where(mask, fmask * bbox[2], feat_ymax)
            feat_xmax = tf.where(mask, fmask * bbox[3], feat_xmax)
            feat_score = tf.where(mask, jaccard, feat_score)
            feat_labels = tf.where(mask, imask * labels[i], feat_labels)
            return [i + 1, feat_ymin, feat_xmin, feat_ymax, feat_xmax, feat_score, feat_labels]

        def condition(i, feat_ymin, feat_xmin, feat_ymax, feat_xmax, feat_score, feat_labels):
            result = tf.less(i, tf.shape(labels))
            return result[0]

        i = 0
        [i, feat_ymin, feat_xmin, feat_ymax, feat_xmax, feat_score, feat_labels] = tf.while_loop(condition, body,
                                                                                                 [i, feat_ymin,
                                                                                                  feat_xmin, feat_ymax,
                                                                                                  feat_xmax, feat_score,
                                                                                                  feat_labels])

        feat_cy = (feat_ymax + feat_ymin) / 2
        feat_cx = (feat_xmax + feat_xmin) / 2
        feat_h = feat_ymax - feat_ymin
        feat_w = feat_xmax - feat_xmin

        feat_cy = (feat_cy - yref) / href
        feat_cx = (feat_cx - xref) / wref
        feat_h = tf.log(tf.div(feat_h, href))
        feat_w = tf.log(tf.div(feat_w, wref))

        feat_loc = tf.stack((feat_cy, feat_cx, feat_h, feat_w), axis=-1)
        return feat_loc, feat_score, feat_labels

    def decode_bboxes(self, ssd_anchors, pred_locs):
        pred_localizations = []
        for anchor, pred_loc in zip(ssd_anchors, pred_locs):
            pred_localization = self.decode_one_bboxes(anchor, pred_loc)
            pred_localizations.append(pred_localization)
        return pred_localizations

    def decode_one_bboxes(self, anchors, pred_loc):
        yref, xref, href, wref = anchors
        cy = pred_loc[:, :, :, :, 0] * href + yref
        cx = pred_loc[:, :, :, :, 1] * wref + xref
        h = tf.exp(pred_loc[:, :, :, :, 2]) * href
        w = tf.exp(pred_loc[:, :, :, :, 3]) * wref

        ymin = cy - h / 2
        xmin = cx - w / 2
        ymax = cy + h / 2
        xmax = cx + w / 2

        pred_localization = tf.stack((xmin, ymin, xmax, ymax), axis=-1)

        return pred_localization

    def interface(self, image, reuse=None, is_training=True):
        end_points = {}
        with tf.variable_scope( 'ssd_300_vgg', reuse=reuse):
            # Original VGG-16 blocks.
            net = slim.repeat(image, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            end_points['block1'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            # Block 2.
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            end_points['block2'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            # Block 3.
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            end_points['block3'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool3',padding='SAME')
            # Block 4.
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            end_points['block4'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool4',padding='SAME')
            # Block 5.
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            end_points['block5'] = net
            net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5',padding='SAME')

            # Additional SSD blocks.
            # Block 6: let's dilate the hell out of it!
            net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
            end_points['block6'] = net
            net = tf.layers.dropout(net, rate=0.5, training=is_training)
            # Block 7: 1x1 conv. Because the fuck.
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
            end_points['block7'] = net
            net = tf.layers.dropout(net, rate=0.5, training=is_training)

            # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
            end_point = 'block8'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
                net = tf.pad(net, [[0,0],[1,1],[1,1],[0,0]])
                net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block9'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = tf.pad(net, [[0,0],[1,1],[1,1],[0,0]])
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block10'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block11'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            end_points[end_point] = net

            # Prediction and localisations layers.
            predictions = []
            logits = []
            localisations = []
            for i, layer in enumerate(self.anchors_blocks):
                with tf.variable_scope(layer + '_box'):
                   l, p = self.ssd_pred(end_points[layer],len(self.anchors_ratios[i])+2)
                predictions.append(slim.softmax(p))
                logits.append(p)
                localisations.append(l)

            return localisations, logits, predictions,

    def ssd_pred(self, net, num_anchors):
        output_shape = utils.tensor_shape(net, 4)[0:-1]
        output_shape += [num_anchors, 4]
        pred_loc = slim.conv2d(net, num_anchors * 4, [3, 3], activation_fn=None,scope='conv_loc')
        pred_loc = tf.reshape(pred_loc, output_shape)


        output_shape = utils.tensor_shape(net, 4)[0:-1] + [num_anchors, self.num_classes]
        pred_logits = slim.conv2d(net, num_anchors * self.num_classes, [3, 3], activation_fn=None,scope='conv_cls')
        pred_logits = tf.reshape(pred_logits, output_shape)
        return pred_loc, pred_logits

    def loss(self, gbboxes, gscores, glabels, pred_loc, pred_logits, hard_mimuming_ratio=3., sess=None):
        fgbboxes = []
        fglabels = []
        fpred_loc = []
        fpred_logits = []
        fgscores = []
        for i in range(len(gbboxes)):
            fgbboxes.append(tf.reshape(gbboxes[i], [-1, 4]))
            fglabels.append(tf.reshape(glabels[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            fpred_loc.append(tf.reshape(pred_loc[i], [-1, 4]))
            fpred_logits.append(tf.reshape(pred_logits[i], [-1, self.num_classes]))
        fgbboxes = tf.concat(fgbboxes, axis=0)
        fglabels = tf.concat(fglabels, axis=0)
        fgscores = tf.concat(fgscores, axis=0)
        fpred_loc = tf.concat(fpred_loc, axis=0)
        fpred_logits = tf.concat(fpred_logits, axis=0)
        fpred_cls = slim.softmax(fpred_logits)

        pmask = fgscores > 0.5
        pmask = tf.logical_and(pmask, fglabels > 0)
        ipmask = tf.cast(pmask, tf.int64)
        fpmask = tf.cast(pmask, tf.float32)
        positive_num = tf.reduce_sum(fpmask)

        negative_num = hard_mimuming_ratio * positive_num + 32
        max_negative_num = tf.reduce_sum(1 - fpmask)
        negative_num = tf.minimum(negative_num, max_negative_num)

        nmask = tf.logical_not(pmask)
        fnmask = tf.cast(nmask, tf.float32)
        negative_values = tf.where(nmask, fpred_cls[:, 0], 1 - fnmask)
        negative_values=tf.reshape(negative_values, [-1])
        n_value, n_index = tf.nn.top_k(-negative_values, tf.to_int32(negative_num))

        n_value = -n_value
        max_neg = tf.reduce_max(n_value)
        neg_mask = tf.logical_and(nmask, negative_values < max_neg)
        neg_fmask = tf.cast(neg_mask, tf.float32)

        losses = 0

        with tf.name_scope('cross_entropy_pos'):

            pos_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fpred_logits, labels=fglabels)

            pos_loss = pos_loss * fpmask
            pos_loss = tf.reduce_sum(pos_loss)
            pos_loss = tf.div(pos_loss, positive_num + 1e-8)
            pos_loss = 1. * pos_loss
            losses += pos_loss
            tf.summary.scalar('cross_entropy_pos_loss', pos_loss)
        with tf.name_scope('cross_entropy_neg'):

            neg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fpred_logits, labels=ipmask)
            neg_loss = neg_loss * neg_fmask
            neg_loss = tf.reduce_sum(neg_loss)
            neg_loss = tf.div(neg_loss, negative_num + 1e-8)
            neg_loss = 10. * neg_loss
            losses += neg_loss
            tf.summary.scalar('cross_entropy_neg_loss', neg_loss)
        with tf.name_scope('localizations_loss'):
            loc_loss = utils.smooth_L1(fgbboxes - fpred_loc)
            weights = tf.expand_dims(fpmask, axis=-1)
            loc_loss = loc_loss * weights
            loc_loss = tf.reduce_sum(loc_loss)
            loc_loss = tf.div(loc_loss, positive_num + 1e-8)
            loc_loss = 5.0 * loc_loss
            losses += loc_loss

            tf.summary.scalar('localization_loss', loc_loss)
        tf.summary.scalar('total_loss', losses)

        return losses, max_neg

    def train_op(self, image, glocalisations, gscores, glabels, sess=None):
        pred_loc, pred_logits, _ = self.interface(image)
        self.losses, num = self.loss(glocalisations, gscores, glabels, pred_loc, pred_logits, sess=sess)
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.losses, global_step=global_step)

        return optimizer, self.losses, global_step,num

    def validate_op(self):
        ssd_anchors = self.place_anchors()
        glocalisations, gscores, glabels = self.encode_bboxes(ssd_anchors, self.bboxes, self.labels)
        pred_loc, pred_logits = self.interface(self.image)
        pred_loc = self.decode_bboxes(ssd_anchors, pred_loc)
        # 只取第一张测试图片
        fpred_loc = []
        fpred_logits = []
        for i in len(pred_loc):
            fpred_loc.append(tf.reshape(pred_loc[0], [-1, 4]))

            fpred_logits.append(tf.reshape(pred_logits[0], [-1, self.num_classes]))
        fpred_loc = tf.concat(fpred_loc, axis=0)
        fpred_logits = tf.concat(fpred_logits, axis=0)
        fpred_scores = slim.softmax(fpred_logits)
        return fpred_loc, fpred_scores, tf.argmax(fpred_scores, axis=-1)

    def test_op(self, image, ssd_anchors, reuse=True, scope='test'):
        pred_loc, pred_logit, pred_cls = self.interface(image, reuse=reuse,is_training=False)
        pred_loc = self.decode_bboxes(ssd_anchors, pred_loc)
        fpred_loc = []
        fpred_scores = []
        fpred_labels = []
        feat_map = []
        fpred_logits = []
        for i in range(len(ssd_anchors)):
            fpred_loc.append(tf.reshape(pred_loc[i][0], [-1, 4]))
            fpred_scores.append(tf.reshape(tf.reduce_max(pred_cls[i][0], axis=-1), [-1]))
            fpred_labels.append(tf.reshape(tf.argmax(pred_cls[i][0], axis=-1), [-1]))
            feat_map.append(tf.reduce_sum(tf.argmax(pred_cls[i][0], axis=-1), axis=-1))
            fpred_logits.append(tf.reshape(pred_logit[i][0], [-1, self.num_classes]))
        fpred_loc = tf.concat(fpred_loc, axis=0)
        fpred_scores = tf.concat(fpred_scores, axis=0)
        fpred_labels = tf.concat(fpred_labels, axis=0)
        fpred_logits = tf.concat(fpred_logits, axis=0)
        pred_mask = tf.greater(fpred_loc, 512)
        pred_fmask = tf.cast(pred_mask, tf.float32)
        fpred_loc = tf.where(fpred_loc < 0, fpred_loc * 0., fpred_loc)
        fpred_loc = tf.where(pred_mask, pred_fmask * 512., fpred_loc)
        # nms
        boxes_index = self.non_max_supression(fpred_loc, fpred_scores, fpred_labels)

        return image[0], fpred_loc, fpred_scores, fpred_labels, boxes_index, fpred_logits

    def non_max_supression(self, pred_loc, pred_scores, pred_labels, max_output_size=20):
        mask = tf.greater(pred_labels, 0)

        mask_scores = tf.where(mask, pred_scores, pred_scores * 0.)

        boxes_index = tf.image.non_max_suppression(pred_loc, mask_scores, max_output_size=max_output_size)

        return boxes_index

    def calc_mAP(self, labels, predictions, k=50):
        return tf.metrics.sparse_average_precision_at_k(labels, predictions, k)



