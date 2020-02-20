import math
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

import detection.shipDetection_ssd.utils.utils as utils
from detection.shipDetection_ssd.config import FLAGS

from basenets.resnet_v2 import resnet_v2_50,resnet_arg_scope


class SSDNet(object):

    def __init__(self):
        self.anchors_blocks=('block1','block2','block3','block4','block5','block6','block7')   #block的索引
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
        self.img_size=512
        self.num_classes=21
        self.localization_prior=[0.1,0.1,0.2,0.2]
        self.losses=0
    def place_anchors(self):
        """
        设置anchors
        :return: 
        """
        ssd_anchors=[]
        for i,sizes in enumerate(self.anchors_sizes):
            anchors=self.place_one_anchors(self.img_size,self.feat_shapes[i],self.anchors_ratios[i]
                                           ,sizes,self.anchors_steps[i],self.offset)
            ssd_anchors.append(anchors)
        return ssd_anchors
    def place_one_anchors(self,img_size,shape,ratios,sizes,step,offset):
        """
        
        :param img_size: 图像的大小
        :param shape: 
        :param ratios: 
        :param sizes: 
        :param step: 
        :param offset: 
        :return: 返回head的锚点以及候选框的大小
        """
        y,x=np.mgrid[0:shape[0],0:shape[1]]
        y=(y.astype(np.float32)+offset)*step/img_size
        x=(x.astype(np.float32)+offset)*step/img_size

        y=tf.expand_dims(y,axis=-1)
        x=tf.expand_dims(x,axis=-1)

        anchors_len=len(ratios)+1
        h=np.zeros((anchors_len,),np.float32)
        w=np.zeros((anchors_len,),np.float32)
        h[0] = np.sqrt(sizes[0]*sizes[1]) / img_size
        w[0] = np.sqrt(sizes[0]*sizes[1]) / img_size
        for i ,ratio in enumerate(ratios):
            h[i+1]=sizes[0]/img_size/math.sqrt(ratio)
            w[i+1]=sizes[0]/img_size*math.sqrt(ratio)

        return y,x,h,w

    def encode_bboxes(self,ssd_anchors,bboxes,labels):
        feat_locs=[]
        feat_scores=[]
        feat_labels=[]
        for i,anchors in enumerate(ssd_anchors):
            loc,score,label=self.encode_one_bboxes(anchors,bboxes,labels)
            feat_locs.append(loc)
            feat_scores.append(score)
            feat_labels.append(label)
        return feat_locs,feat_scores,feat_labels
    def encode_one_bboxes(self,anchors,bboxes,labels,threshold=0.5):
        yref,xref,href,wref=anchors
        ymin=yref-href/2
        xmin=xref-wref/2
        ymax=yref+href/2
        xmax=xref+wref/2
        anchors_area=(ymax-ymin)*(xmax-xmin)

        anchor_shape=np.shape(ymin)
        feat_ymin=tf.zeros(anchor_shape,tf.float32)
        feat_xmin=tf.zeros(anchor_shape,tf.float32)
        feat_ymax=tf.ones(anchor_shape,tf.float32)
        feat_xmax=tf.ones(anchor_shape,tf.float32)
        feat_score=tf.zeros(anchor_shape,tf.float32)
        feat_labels=tf.zeros(anchor_shape,tf.int64)

        def jaccard_similarity(bbox):
            """
            计算anchors与任意bbox的jaccard相似度
            :param bbox: 
            :return: 返回交并比
            """
            inter_ymin=tf.maximum(ymin,bbox[0])
            inter_xmin=tf.maximum(xmin,bbox[1])
            inter_ymax=tf.minimum(ymax,bbox[2])
            inter_xmax=tf.minimum(xmax,bbox[3])

            inter_h=tf.maximum(inter_ymax-inter_ymin,0.)
            inter_w=tf.maximum(inter_xmax-inter_xmin,0.)
            inter_area=inter_h*inter_w
            total_area=anchors_area-inter_area+(bbox[3]-bbox[1])*(bbox[2]-bbox[0])
            return tf.div(inter_area,total_area)

        def body(i,feat_ymin,feat_xmin,feat_ymax,feat_xmax,feat_score,feat_labels):
            bbox=bboxes[i]

            jaccard=jaccard_similarity(bbox)
            mask=tf.greater(jaccard,feat_score)
            mask=tf.logical_and(mask,tf.greater_equal(jaccard,threshold))
            imask=tf.cast(mask,tf.int64)
            fmask=tf.cast(mask,tf.float32)
            feat_ymin=tf.where(mask,fmask*bbox[0],feat_ymin)
            feat_xmin=tf.where(mask,fmask*bbox[1],feat_xmin)
            feat_ymax=tf.where(mask,fmask*bbox[2],feat_ymax)
            feat_xmax=tf.where(mask,fmask*bbox[3],feat_xmax)
            feat_score=tf.where(mask,jaccard,feat_score)
            feat_labels=tf.where(mask,imask*labels[i],feat_labels)
            return [i+1,feat_ymin,feat_xmin,feat_ymax,feat_xmax,feat_score,feat_labels]

        def condition(i,feat_ymin,feat_xmin,feat_ymax,feat_xmax,feat_score,feat_labels):
            result=tf.less(i,tf.shape(labels))
            return result[0]
        i=0
        [i,feat_ymin,feat_xmin,feat_ymax,feat_xmax,feat_score,feat_labels]=tf.while_loop(condition,body,
                                                    [i,feat_ymin,feat_xmin,feat_ymax,feat_xmax,feat_score,feat_labels])

        feat_cy=(feat_ymax+feat_ymin)/2
        feat_cx=(feat_xmax+feat_xmin)/2
        feat_h=feat_ymax-feat_ymin
        feat_w=feat_xmax-feat_xmin

        feat_cy=(feat_cy-yref)/href/self.localization_prior[0]
        feat_cx=(feat_cx-xref)/wref/self.localization_prior[1]
        feat_h=tf.log(tf.div(feat_h,href))/self.localization_prior[2]
        feat_w=tf.log(tf.div(feat_w,wref))/self.localization_prior[3]

        feat_loc=tf.stack((feat_cy,feat_cx,feat_h,feat_w),axis=-1)
        return feat_loc,feat_score,feat_labels

    def decode_bboxes(self,ssd_anchors,pred_locs):
        pred_localizations=[]
        for anchor,pred_loc in zip(ssd_anchors,pred_locs):
            pred_localization=self.decode_one_bboxes(anchor,pred_loc)
            pred_localizations.append(pred_localization)
        return pred_localizations
    def decode_one_bboxes(self,anchors,pred_loc):
         yref,xref,href,wref=anchors
         cy=pred_loc[:,:,:,:,0]*href*self.localization_prior[0]+yref
         cx=pred_loc[:,:,:,:,1]*wref*self.localization_prior[1]+xref
         h=tf.exp(pred_loc[:,:,:,:,2]*self.localization_prior[2])*href
         w=tf.exp(pred_loc[:,:,:,:,3]*self.localization_prior[3])*wref

         ymin=cy-h/2
         xmin=cx-w/2
         ymax=cy+h/2
         xmax=cx+w/2

         pred_localization=tf.stack((ymin,xmin,ymax,xmax),axis=-1)

         return pred_localization

    def interface(self,image,reuse=None,is_training=True):
        endpoints={}
        arg_scope=resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            _,resnet_endpoints=resnet_v2_50(image,1001,is_training=is_training,reuse=reuse)
        with tf.variable_scope('ssd_fpn',reuse=reuse):
            net_block1=resnet_endpoints['resnet_v2_50/block1/unit_3/bottleneck_v2/conv3']
            # endpoints['block1']=net_block1
            net_block2=resnet_endpoints['resnet_v2_50/block2/unit_4/bottleneck_v2/conv3']
            # endpoints['block2']=net_block2
            net_block3=resnet_endpoints['resnet_v2_50/block3/unit_6/bottleneck_v2/conv3']
            # endpoints['block3']=net_block3
            net_block4=resnet_endpoints['resnet_v2_50/block4/unit_3/bottleneck_v2/conv3']
            net_block4=slim.conv2d(net_block4,1024,[3,3],stride=2,weights_regularizer=slim.l2_regularizer(0.0005))


            net_block5=slim.conv2d(net_block4,1024,[1,1],stride=1,weights_regularizer=slim.l2_regularizer(0.0005))
            net_block5=slim.conv2d(net_block5,512,[3,3],stride=2,weights_regularizer=slim.l2_regularizer(0.0005))
            endpoints['block5']=net_block5
            net_block4+=slim.conv2d_transpose(net_block5,1024,[3,3],stride=2)
            endpoints['block4'] = net_block4
            net_block3+=slim.conv2d_transpose(net_block4,1024,[3,3],stride=2)
            endpoints['block3'] = net_block3
            net_block2 += slim.conv2d_transpose(net_block3, 512, [3, 3], stride=2,weights_regularizer=slim.l2_regularizer(0.0005))
            endpoints['block2'] = net_block2
            net_block1 += slim.conv2d_transpose(net_block2, 256, [3, 3], stride=2,weights_regularizer=slim.l2_regularizer(0.0005))
            endpoints['block1'] = net_block1
            net_block6 = slim.conv2d(net_block5, 128, [1, 1], stride=1,weights_regularizer=slim.l2_regularizer(0.0005))
            net_block6 = slim.conv2d(net_block6, 256, [3, 3], stride=2,weights_regularizer=slim.l2_regularizer(0.0005))
            endpoints['block6'] = net_block6

            net_block7 = slim.conv2d(net_block6, 128, [1, 1], stride=1,weights_regularizer=slim.l2_regularizer(0.0005))
            net_block7 = slim.conv2d(net_block7, 256, [2, 2], stride=1,padding='VALID')
            endpoints['block7'] = net_block7

        with tf.variable_scope('pred',reuse=reuse):
            pred_loc=[]
            pred_logits=[]
            pred_cls=[]
            for i,block in enumerate(self.anchors_blocks):
                net=endpoints[block]
                loc,logits=self.ssd_pred(net,len(self.anchors_ratios[i])+1)
                cls=slim.softmax(logits)
                pred_loc.append(loc)
                pred_logits.append(logits)
                pred_cls.append(cls)
        return pred_loc,pred_logits,pred_cls
    def ssd_pred(self,net,num_anchors):
        with tf.name_scope('pred_loc'):
            output_shape=utils.tensor_shape(net,4)[0:-1]
            output_shape+=[num_anchors,4]
            pred_loc=slim.conv2d(net,num_anchors*4,[3,3],activation_fn=None)
            pred_loc=tf.reshape(pred_loc,output_shape)

        with tf.name_scope('pred_cls'):
            output_shape=utils.tensor_shape(net,4)[0:-1]+[num_anchors,self.num_classes]
            pred_logits=slim.conv2d(net,num_anchors*self.num_classes,[3,3],activation_fn=None)
            pred_logits=tf.reshape(pred_logits,output_shape)
        return pred_loc,pred_logits

    def loss(self,gbboxes,gscores,glabels,pred_loc,pred_logits,hard_mimuming_ratio=3.,sess=None):
        fgbboxes=[]
        fglabels=[]
        fpred_loc=[]
        fpred_logits=[]
        fgscores=[]
        for i in range(len(gbboxes)):
            fgbboxes.append(tf.reshape(gbboxes[i],[-1,4]))
            fglabels.append(tf.reshape(glabels[i],[-1]))
            fgscores.append(tf.reshape(gscores[i],[-1]))
            fpred_loc.append(tf.reshape(pred_loc[i],[-1,4]))
            fpred_logits.append(tf.reshape(pred_logits[i],[-1,self.num_classes]))
        fgbboxes=tf.concat(fgbboxes,axis=0)
        fglabels=tf.concat(fglabels,axis=0)
        fgscores=tf.concat(fgscores,axis=0)
        fpred_loc=tf.concat(fpred_loc,axis=0)
        fpred_logits=tf.concat(fpred_logits,axis=0)
        fpred_cls=slim.softmax(fpred_logits)

        pmask=fgscores>0.5
        pmask=tf.logical_and(pmask,fglabels>0)
        ipmask=tf.cast(pmask,tf.int64)
        fpmask=tf.cast(pmask,tf.float32)
        positive_num=tf.reduce_sum(fpmask)

        negative_num=hard_mimuming_ratio*positive_num+20
        max_negative_num=tf.reduce_sum(1-fpmask)
        negative_num=tf.minimum(negative_num,max_negative_num)

        nmask=tf.logical_not(pmask)
        fnmask=tf.cast(nmask,tf.float32)
        negative_values=tf.where(nmask,fpred_cls[:,0],1-fnmask)
        # negative_values=tf.reshape(-negative_values, [-1])
        n_value,n_index=tf.nn.top_k(-negative_values,tf.to_int32(negative_num))
        n_value=-n_value
        max_neg=tf.reduce_max(n_value)
        neg_mask=tf.logical_and(nmask,negative_values<max_neg)
        neg_fmask=tf.cast(neg_mask,tf.float32)

        losses=0
        with tf.name_scope('cross_entropy_pos'):
            pos_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fpred_logits,labels=fglabels)

            pos_loss=pos_loss*fpmask
            pos_loss=tf.reduce_sum(pos_loss)
            pos_loss=tf.div(pos_loss,positive_num+1e-8)
            pos_loss=1.*pos_loss
            losses+=pos_loss
            tf.summary.scalar('cross_entropy_pos_loss',pos_loss)
        with tf.name_scope('cross_entropy_neg'):
            neg_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fpred_logits,labels=ipmask)
            neg_loss=neg_loss*neg_fmask
            neg_loss=tf.reduce_sum(neg_loss)
            neg_loss=tf.div(neg_loss,negative_num+1e-8)
            neg_loss = 1. * neg_loss
            losses+=neg_loss
            tf.summary.scalar('cross_entropy_neg_loss', neg_loss)
        with tf.name_scope('localizations_loss'):
            loc_loss=utils.smooth_L1(fgbboxes-fpred_loc)
            weights=tf.expand_dims(fpmask,axis=-1)
            loc_loss=loc_loss*weights
            loc_loss=tf.reduce_sum(loc_loss)
            loc_loss=tf.div(loc_loss,positive_num+1e-8)
            loc_loss=1.0*loc_loss
            losses+=loc_loss

            tf.summary.scalar('localization_loss', loc_loss)
        tf.summary.scalar('total_loss', losses)

        return losses,max_neg

    def train_op(self,image,glocalisations,gscores,glabels,sess=None):
        pred_loc, pred_logits,_=self.interface(image)
        self.losses,num=self.loss(glocalisations,gscores,glabels,pred_loc,pred_logits,sess=sess)
        global_step=tf.train.get_or_create_global_step()
        optimizer=tf.train.AdamOptimizer(0.0001).minimize(self.losses,global_step=global_step)

        return optimizer,self.losses,global_step,num






    def test_op(self,image,ssd_anchors,reuse=True,scope='test'):
        pred_loc,pred_logit,pred_cls=self.interface(image,reuse=reuse)
        pred_loc=self.decode_bboxes(ssd_anchors,pred_loc)
        fpred_loc=[]
        fpred_scores=[]
        fpred_labels=[]
        feat_map=[]
        fpred_logits=[]
        for i in range(len(ssd_anchors)):
            fpred_loc.append(tf.reshape(pred_loc[i][0],[-1,4]))
            fpred_scores.append(tf.reshape(tf.reduce_max(pred_cls[i][0],axis=-1),[-1]))
            fpred_labels.append(tf.reshape(tf.argmax(pred_cls[i][0],axis=-1),[-1]))
            feat_map.append(tf.reduce_sum(tf.argmax(pred_cls[i][0],axis=-1),axis=-1))
            fpred_logits.append(tf.reshape(pred_logit[i][0],[-1,self.num_classes]))
        fpred_loc=tf.concat(fpred_loc,axis=0)

        fpred_scores=tf.concat(fpred_scores,axis=0)
        fpred_labels=tf.concat(fpred_labels,axis=0)
        fpred_logits=tf.concat(fpred_logits,axis=0)
        pred_mask=tf.greater(fpred_loc,1)
        pred_fmask=tf.cast(pred_mask,tf.float32)
        fpred_loc=tf.where(fpred_loc<0,fpred_loc*0.,fpred_loc)
        fpred_loc=tf.where(pred_mask,pred_fmask*1.,fpred_loc)
        #nms
        boxes_index=self.non_max_supression(fpred_loc,fpred_scores,fpred_labels)

        print(boxes_index)
        return [image[0],fpred_loc,fpred_scores,fpred_labels,boxes_index,fpred_logits]

    def non_max_supression(self,pred_loc,pred_scores,pred_labels,max_output_size=20):
        mask=tf.greater(pred_labels,0)

        mask_scores=tf.where(mask,pred_scores,pred_scores*0.)

        boxes_index=tf.image.non_max_suppression(pred_loc,mask_scores,max_output_size=max_output_size)

        return boxes_index




