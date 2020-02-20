import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils.utils as utils
from nets.vgg import vgg_16,vgg_arg_scope
from config import FLAGS
class SSDNet_vgg_512(object):
    def __init__(self):
        self.anchors_blocks = ['block4', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12']  # block的索引
        self.anchors_steps = (8, 16, 32, 64, 128, 256, 512)
        self.offset = 0.5
        self.anchors_ratios = [[2, .5, 1],
                               [2, .5, 3, 1. / 3, 1],
                               [2, .5, 3, 1. / 3, 1],
                               [2, .5, 3, 1. / 3, 1],
                               [2, .5, 3, 1. / 3, 1],
                               [2, .5, 1],
                               [2, .5, 1]]
        self.anchors_sizes = [(20.48, 51.2),
                              (51.2, 133.12),
                              (133.12, 215.04),
                              (215.04, 296.96),
                              (296.96, 378.88),
                              (378.88, 460.8),
                              (460.8, 542.72)]
        self.feat_shapes = [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]
        self.img_size = 512
        self.num_classes = 4
        self.localization_prior = [0.1, 0.1, 0.2, 0.2]
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

        anchors_len = len(ratios) + 1
        h = np.zeros((anchors_len,), np.float32)
        w = np.zeros((anchors_len,), np.float32)
        h[0] = np.sqrt(sizes[0] * sizes[1]) / img_size
        w[0] = np.sqrt(sizes[0] * sizes[1]) / img_size
        for i, ratio in enumerate(ratios):
            h[i + 1] = sizes[0] / img_size / math.sqrt(ratio)
            w[i + 1] = sizes[0] / img_size * math.sqrt(ratio)

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

        feat_cy = (feat_cy - yref) / href / self.localization_prior[0]
        feat_cx = (feat_cx - xref) / wref / self.localization_prior[1]
        feat_h = tf.log(tf.div(feat_h, href)) / self.localization_prior[2]
        feat_w = tf.log(tf.div(feat_w, wref)) / self.localization_prior[3]

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
        cy = pred_loc[:, :, :, :, 0] * href * self.localization_prior[0] + yref
        cx = pred_loc[:, :, :, :, 1] * wref * self.localization_prior[1] + xref
        h = tf.exp(pred_loc[:, :, :, :, 2] * self.localization_prior[2]) * href
        w = tf.exp(pred_loc[:, :, :, :, 3] * self.localization_prior[3]) * wref

        ymin = cy - h / 2
        xmin = cx - w / 2
        ymax = cy + h / 2
        xmax = cx + w / 2

        pred_localization = tf.stack((ymin, xmin, ymax, xmax), axis=-1)

        return pred_localization

    def interface(self, inputs, reuse=None, is_training=True,scope='ssd_512_vgg'):

        end_points = {}

            # Original VGG-16 blocks.
        with slim.arg_scope(vgg_arg_scope()):
            _,vgg_end_points=vgg_16(inputs,is_training=is_training,reuse=reuse,
                                    spatial_squeeze=False,num_classes=None)

            end_points['block1'] = vgg_end_points['vgg_16/conv1/conv1_2']

            end_points['block2'] = vgg_end_points['vgg_16/conv2/conv2_2']

            end_points['block3'] = vgg_end_points['vgg_16/conv3/conv3_3']

            end_points['block4'] = vgg_end_points['vgg_16/conv4/conv4_3']

            end_points['block5'] = vgg_end_points['vgg_16/conv5/conv5_3']

        with tf.variable_scope(scope, 'ssd_512_vgg', [inputs], reuse=reuse):
            # Additional SSD blocks.
            # Block 6: let's dilate the hell out of it!
            net=end_points['block5']
            net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
            end_points['block6'] = net
            # Block 7: 1x1 conv. Because the fuck.
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
            end_points['block7'] = net
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
                net = tf.pad(net, [[0,0],[1,1],[1,1],[0,0]])
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block11'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = tf.pad(net, [[0,0],[1,1],[1,1],[0,0]])
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block12'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = tf.pad(net, [[0,0],[1,1],[1,1],[0,0]])
                net = slim.conv2d(net, 256, [4, 4], scope='conv4x4', padding='VALID')
            end_points[end_point] = net

        # with tf.variable_scope(scope, 'ssd_512_attention_branch', [inputs], reuse=reuse):
        #     net=end_points['block5']
        #     net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='convF_6')
        #     end_points['attention_1'] = net
        #     # Block 7: 1x1 conv. Because the fuck.
        #     net = slim.conv2d(net, 1024, [1, 1], scope='convF_7')
        #     end_points['attention_2'] = net
        #     # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        #     end_point = 'attention_3'
        #     with tf.variable_scope(end_point):
        #         net = slim.conv2d(net, 256, [1, 1], scope='convF_1x1')
        #         net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
        #         net = slim.conv2d(net, 512, [3, 3], stride=2, scope='convF_3x3', padding='VALID')
        #     end_points[end_point] = net
        #     end_point = 'attention_4'
        #     with tf.variable_scope(end_point):
        #         net = slim.conv2d(net, 128, [1, 1], scope='convF_1x1')
        #         net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
        #         net = slim.conv2d(net, 256, [3, 3], stride=2, scope='convF_3x3', padding='VALID')
        #     end_points[end_point] = net
        #     mask_logits = slim.conv2d(net, 2, [3, 3], stride=1, scope='convF_3x3', padding='SAME')
        #     mask_pre = slim.softmax(mask_logits)
        #
        #     with tf.variable_scope(scope, 'ssd_512_fuse', [inputs], reuse=reuse):
        #         net = end_points['attention_4']
        #
        #         deconv_shape1 = end_points['block6'].get_shape()
        #         W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, 256], name="W_t1")
        #         b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        #         conv_t1 = utils.conv2d_transpose_strided(net, W_t1, b_t1, output_shape=tf.shape(end_points['block6']))
        #         fuse_1 = tf.add(conv_t1, end_points['block6'], name="fuse_1")
        #         end_points['block6'] = fuse_1
        #
        #         deconv_shape2 = end_points['block5'].get_shape()
        #         W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, 1024], name="W_t2")
        #         b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        #         conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(end_points['block5']))
        #         fuse_2 = tf.add(conv_t2, end_points['block5'], name="fuse_2")
        #         end_points['block5'] = fuse_2
        #
        #         conv_t3=slim.conv2d_transpose(fuse_2,128,[16,16],stride=8)
        #         fuse_3 = tf.add(conv_t3, end_points['block2'], name="fuse_3")
        #         end_points['block2'] = fuse_3
        #
        #     print(end_points)

        pred_loc = []
        pred_logits = []
        pred_cls = []
        for i, block in enumerate(self.anchors_blocks):
            with tf.variable_scope(block + '_box',reuse=reuse):
                net = end_points[block]
                loc, logits = self.ssd_pred(net, len(self.anchors_ratios[i]) + 1)
                cls = slim.softmax(logits)
                pred_loc.append(loc)
                pred_logits.append(logits)
                pred_cls.append(cls)
        return pred_loc, pred_logits, pred_cls

    def ssd_pred(self, net, num_anchors):

        output_shape = utils.tensor_shape(net, 4)[0:-1]
        output_shape += [num_anchors, 4]
        pred_loc = slim.conv2d(net, num_anchors * 4, [3, 3], activation_fn=None,scope='conv_loc')
        pred_loc = tf.reshape(pred_loc, output_shape)

        output_shape = utils.tensor_shape(net, 4)[0:-1] + [num_anchors, self.num_classes]
        pred_logits = slim.conv2d(net,num_anchors * self.num_classes, [3, 3], activation_fn=None,scope='conv_cls')
        pred_logits = tf.reshape(pred_logits, output_shape)
        return pred_loc, pred_logits

    def loss_loc(self, mask_pre, mask):
        print(mask)

        mask=tf.one_hot(mask,depth=2,axis=-1)
        print(mask)
        mask = tf.cast(mask, tf.float32)
        loss_loc = tf.reduce_mean((mask-mask_pre)**2)
        loss_location = tf.reduce_mean(loss_loc)

        return loss_location

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

        negative_num = hard_mimuming_ratio * positive_num + 20
        max_negative_num = tf.reduce_sum(1 - fpmask)
        negative_num = tf.minimum(negative_num, max_negative_num)

        nmask = tf.logical_not(pmask)
        fnmask = tf.cast(nmask, tf.float32)
        negative_values = tf.where(nmask, fpred_cls[:, 0], 1 - fnmask)
        # negative_values=tf.reshape(-negative_values, [-1])
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
            neg_loss = 1. * neg_loss
            losses += neg_loss
            tf.summary.scalar('cross_entropy_neg_loss', neg_loss)
        with tf.name_scope('localizations_loss'):
            loc_loss = utils.smooth_L1(fgbboxes - fpred_loc)
            weights = tf.expand_dims(fpmask, axis=-1)
            loc_loss = loc_loss * weights
            loc_loss = tf.reduce_sum(loc_loss)
            loc_loss = tf.div(loc_loss, positive_num + 1e-8)
            loc_loss = 1.0 * loc_loss
            losses += loc_loss

            tf.summary.scalar('localization_loss', loc_loss)
        tf.summary.scalar('total_loss', losses)

        return losses, max_neg

    def train_op(self, image, glocalisations, gscores, glabels, sess=None):
        with slim.arg_scope(self.ssd_arg_scope()):
            pred_loc, pred_logits, _  = self.interface(image)

        self.losses, num = self.loss(glocalisations, gscores, glabels, pred_loc, pred_logits, sess=sess)

        global_step = tf.train.get_or_create_global_step()

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.losses, global_step=global_step)

        return optimizer, self.losses, global_step, num

    def validate_op(self, image, glocalisations, gscores, glabels,ssd_anchors,reuse=None):
        with slim.arg_scope(self.ssd_arg_scope()):
            pred_loc, pred_logit, pred_cls  = self.interface(image,reuse=reuse)

        self.losses, num = self.loss(glocalisations, gscores, glabels, pred_loc, pred_logit)

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
        pred_mask = tf.greater(fpred_loc, 1)
        pred_fmask = tf.cast(pred_mask, tf.float32)
        fpred_loc = tf.where(fpred_loc < 0, fpred_loc * 0., fpred_loc)
        fpred_loc = tf.where(pred_mask, pred_fmask * 1., fpred_loc)
        # nms
        boxes_index = self.non_max_supression(fpred_loc, fpred_scores, fpred_labels,20)

        return [image[0], fpred_loc, fpred_scores, fpred_labels, boxes_index, fpred_logits,self.losses]

    def test_op(self, image, ssd_anchors, reuse=True, scope='test'):
        with slim.arg_scope(self.ssd_arg_scope()):
            pred_loc, pred_logit, pred_cls= self.interface(image, reuse=reuse)

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
        pred_mask = tf.greater(fpred_loc, 1)
        pred_fmask = tf.cast(pred_mask, tf.float32)
        fpred_loc = tf.where(fpred_loc < 0, fpred_loc * 0., fpred_loc)
        fpred_loc = tf.where(pred_mask, pred_fmask * 1., fpred_loc)
        # nms
        boxes_index = self.non_max_supression(fpred_loc, fpred_scores, fpred_labels, 200)


        return [image[0], fpred_loc, fpred_scores, fpred_labels, boxes_index, fpred_logits]

    def non_max_supression(self, pred_loc, pred_scores, pred_labels, max_output_size=20):
        mask = tf.greater(pred_labels, 0)

        mask_scores = tf.where(mask, pred_scores, pred_scores * 0.)

        boxes_index = tf.image.non_max_suppression(pred_loc, mask_scores, max_output_size=max_output_size)

        return boxes_index

    def calc_mAP(self, labels, predictions, k=50):
        return tf.metrics.sparse_average_precision_at_k(labels, predictions, k)

    def ssd_arg_scope(self,weight_decay=0.0005, data_format='NHWC'):
        """Defines the VGG arg scope.

        Args:
          weight_decay: The l2 regularization coefficient.

        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME',
                                data_format=data_format) as sc:

                return sc
