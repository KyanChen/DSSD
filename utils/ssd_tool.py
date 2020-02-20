#anchor设置的数量
import numpy as np
import cv2
import math
import dataset.createTFRecords as utils
import os
import glob
import matplotlib.pylab as plt
import uuid
class_names=['background','areplane','ship','tank']
class_dict=dict(zip(range(0,len(class_names)),class_names))
print(class_dict)
class ssdTool():
    def __init__(self):
        self.anchors_blocks = ['block1', 'block2', 'block3', 'block4', 'block7', 'block8', 'block9', 'block10',
                               'block11', 'block12']  # block的索引
        self.anchors_steps = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
        self.offset = 0.5
        self.anchors_ratios = [[2, .5, 1],
                               [2, .5, 1],
                               [2, .5, 1],
                               [2, .5, 3, 1. / 3, 1],
                               [2, .5, 3, 1. / 3, 1],
                               [2, .5, 3, 1. / 3, 1],
                               [2, .5, 3, 1. / 3, 1],
                               [2, .5, 1],
                               [2, .5, 1]]
        self.anchors_sizes = [(8, 8),
                              (10, 15),
                              (20.48, 51.2),
                              (51.2, 133.12),
                              (133.12, 215.04),
                              (215.04, 296.96),
                              (296.96, 378.88),
                              (378.88, 460.8),
                              (460.8, 542.72)]

        self.feat_shapes = [(256, 256), (128, 128), (64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]
        self.img_size = 512
        self.num_classes = 2
        self.localization_prior = [0.1, 0.1, 0.2, 0.2]
        self.img_size=512
        self.bboxes_dict={}
        self.class_dict=dict(zip(range(0,len(class_names)),class_names))
        self.labels_dict={}
        self.num_boxes=0
        self.catched_boxes={}
    def get_dict_key(self,height,width):
        return '%4.f*%4.f '%(height*512,width*512)

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

        y = np.expand_dims(y, axis=-1)
        x = np.expand_dims(x, axis=-1)

        anchors_len = len(ratios)
        h = np.zeros((anchors_len,), np.float32)
        w = np.zeros((anchors_len,), np.float32)
        for i, ratio in enumerate(ratios):
            h[i] = sizes[0] / img_size / math.sqrt(ratio)
            w[i] = sizes[1] / img_size * math.sqrt(ratio)
            self.bboxes_dict[self.get_dict_key(h[i],w[i])]=0
        return y, x, h, w

    def bboxes_Histogram(self,bboxes,ssd_anchors,labels):
        for anchors in ssd_anchors:
            self.bboxes_one_histogram(anchors,bboxes,labels)
    def bboxes_one_histogram(self,anchors,bboxes,labels):
        def jaccard_similirity(ymin,xmin,ymax,xmax,bbox):
            inter_ymin=np.maximum(ymin,bbox[0])
            inter_xmin=np.maximum(xmin,bbox[1])
            inter_ymax=np.minimum(ymax,bbox[2])
            inter_xmax=np.minimum(xmax,bbox[3])
            h=np.maximum(inter_ymax-inter_ymin,0)
            w=np.maximum(inter_xmax-inter_xmin,0)
            inter_area=h*w
            area=(ymax-ymin)*(xmax-xmin)
            return inter_area/(area-inter_area+(bbox[3]-bbox[1])*(bbox[2]-bbox[0]))
        def count_bboxes(ymin,xmin,ymax,xmax,bboxes):
            count=0
            for i,bbox in enumerate(bboxes):
                jaccard=jaccard_similirity(ymin,xmin,ymax,xmax,bbox)
                mask=jaccard>0.5
                imask=mask.astype(np.int)
                temp=np.sum(imask)
                self.num_boxes+=temp
                count+=np.sum(imask)
                labels_key=self.class_dict[labels[i]]
                if labels_key in self.labels_dict.keys():
                    self.labels_dict[labels_key]+=np.sum(imask)
                else:
                    self.labels_dict[labels_key]=0
            return count
        yref,xref,href,wref=anchors
        for i in range(len(href)):
            ymin=yref-href[i]/2
            xmin=xref-wref[i]/2
            ymax=yref+href[i]/2
            xmax=xref+wref[i]/2
            dict_key=self.get_dict_key(href[i],wref[i])
            self.bboxes_dict[dict_key]+=count_bboxes(ymin,xmin,ymax,xmax,bboxes)

    def encode_bboxes(self,ssd_anchors,bboxes,labels,img_name=None):
        feat_locs=[]
        feat_scores=[]
        feat_labels=[]
        for i,anchors in enumerate(ssd_anchors):
            loc,score,label=self.encode_one_bboxes(anchors,bboxes,labels,img_name=img_name)
            feat_locs.append(loc)
            feat_scores.append(score)
            feat_labels.append(label)
        return feat_locs,feat_scores,feat_labels
    def encode_one_bboxes(self,anchors,bboxes,labels,threshold=0.5,img_name=None):
        yref,xref,href,wref=anchors
        ymin=yref-href/2
        xmin=xref-wref/2
        ymax=yref+href/2
        xmax=xref+wref/2
        anchors_area=(ymax-ymin)*(xmax-xmin)

        anchor_shape=np.shape(ymin)

        feat_ymin=np.zeros(anchor_shape,np.float32)
        feat_xmin=np.zeros(anchor_shape,np.float32)
        feat_ymax=np.ones(anchor_shape,np.float32)
        feat_xmax=np.ones(anchor_shape,np.float32)
        feat_score=np.zeros(anchor_shape,np.float32)
        feat_labels=np.zeros(anchor_shape,np.int64)

        def jaccard_similarity(bbox):
            """
            计算anchors与任意bbox的jaccard相似度
            :param bbox: 
            :return: 返回交并比
            """
            inter_ymin=np.maximum(ymin,bbox[0])
            inter_xmin=np.maximum(xmin,bbox[1])
            inter_ymax=np.minimum(ymax,bbox[2])
            inter_xmax=np.minimum(xmax,bbox[3])

            inter_h=np.maximum(inter_ymax-inter_ymin,0.)
            inter_w=np.maximum(inter_xmax-inter_xmin,0.)
            inter_area=inter_h*inter_w
            total_area=anchors_area-inter_area+(bbox[3]-bbox[1])*(bbox[2]-bbox[0])
            return inter_area/total_area



        def body(i, feat_ymin, feat_xmin, feat_ymax, feat_xmax, feat_score, feat_labels):
            bbox = bboxes[i]
            jaccard=jaccard_similarity(bbox)

            mask=np.greater(jaccard,feat_score)
            mask=np.logical_and(mask,np.greater_equal(jaccard,threshold))
            imask=np.float32(mask)
            fmask=np.float32(mask)
            key='%s_%d_%dx%d'%(img_name,i,(bbox[2]-bbox[0])*512,(bbox[3]-bbox[1])*512)
            if np.sum(imask)>0:
                if key not in self.catched_boxes.keys():
                    self.catched_boxes[key]=1
                else:
                    self.catched_boxes[key]+=1
            else:
                if key not in self.catched_boxes.keys():
                    self.catched_boxes[key] = 0

            feat_ymin=np.where(mask,fmask*ymin,feat_ymin)
            feat_xmin=np.where(mask,fmask*xmin,feat_xmin)
            feat_ymax=np.where(mask,fmask*ymax,feat_ymax)
            feat_xmax=np.where(mask,fmask*xmax,feat_xmax)
            feat_score=np.where(mask,jaccard,feat_score)
            feat_labels=np.where(mask,imask*labels[i],feat_labels)
            return [i+1,feat_ymin,feat_xmin,feat_ymax,feat_xmax,feat_score,feat_labels]

        def condition(i,feat_ymin,feat_xmin,feat_ymax,feat_xmax,feat_score,feat_labels):
            result=np.less(i,np.shape(labels))
            return result[0]
        i=0
        while condition(i,feat_ymin,feat_xmin,feat_ymax,feat_xmax,feat_score,feat_labels):
            [i,feat_ymin,feat_xmin,feat_ymax,feat_xmax,feat_score,feat_labels]=body(i,feat_ymin,feat_xmin,feat_ymax,
                                                                                     feat_xmax,feat_score,feat_labels)
        feat_ymin=np.int32(feat_ymin*512)
        feat_xmin=np.int32(feat_xmin*512)
        feat_ymax=np.int32(feat_ymax*512)
        feat_xmax=np.int32(feat_xmax*512)




        feat_loc=np.stack((feat_ymin,feat_xmin,feat_ymax,feat_xmax),axis=-1)

        return feat_loc,feat_score,feat_labels

    def crop_imgage(self,img,feat_loc,feat_labels,feat_scores,imgPath):
        ffeat_loc=[]
        ffeat_labels=[]
        ffeat_scores=[]
        for i in range(len(feat_loc)):
            ffeat_loc.append(np.reshape(feat_loc[i],(-1,4)))
            ffeat_labels.append(np.reshape(feat_labels[i],(-1)))
            ffeat_scores.append(np.reshape(feat_scores[i],(-1)))

        ffeat_loc=np.concatenate(ffeat_loc,0)
        ffeat_labels=np.concatenate(ffeat_labels,0)
        ffeat_scores=np.concatenate(ffeat_scores,0)
        print(np.shape(img))
        img=cv2.resize(img,(512,512))
        crop_img_path=os.path.join('crop_image',imgPath.replace('.jpg',''))
        if not os.path.exists(crop_img_path):
            os.mkdir(crop_img_path)
        for i in range(len(ffeat_scores)):
            if ffeat_labels[i] in class_dict.keys():
                labels=class_dict[ffeat_labels[i]]
            else:
                labels='background'
            crop_img_per_path=os.path.join(crop_img_path,labels)
            if not os.path.exists(crop_img_per_path):
                os.mkdir(crop_img_per_path)
            crop_img_filename=os.path.join(crop_img_per_path,'%d_%f.jpg'%(i,ffeat_scores[i]))
            loc=ffeat_loc[i,:]

            area_h=(loc[2]-loc[0])
            area_w=loc[3]-loc[1]
            area_h=np.maximum(area_h,0.)
            area_w=np.maximum(area_w,0.)
            area=area_h*area_w
            if area>0 and ffeat_labels[i]>0:
                # print(loc)
                crop_img=img[loc[0]:loc[2],loc[1]:loc[3]]
                cv2.imwrite(crop_img_filename,crop_img)


def calc_bndboxes(bndboxes,labels,boxes_dict,labels_dict):
    class_dict = dict(zip(range(0, len(class_names)), class_names))
    for label,box in zip(labels,bndboxes):
        label=class_dict[label]
        boxes_key=(box[3]-box[1])*(box[2]-box[0])
        boxes_key=boxes_key*512*512
        boxes_key=math.sqrt(boxes_key)//20
        if label in labels_dict.keys():
            labels_dict[label]+=1
        else:
            labels_dict[label]=1
        if boxes_key in boxes_dict.keys():
            boxes_dict[boxes_key]+=1
        else:
            boxes_dict[boxes_key]=1
if __name__=='__main__':
    ssd_tool=ssdTool()
    dataset_path = r'dataset\train_data'
    label_path = r'dataset\train_data'

    txtfiles=glob.glob(os.path.join(label_path,'*.txt'))
    txtfiles=[x for x in txtfiles if os.path.getsize(x)>0]
    ssd_anchors=ssd_tool.place_anchors()
    count=0
    i=0
    labels_dict={}
    boxes_dict={}
    for i,file in enumerate(txtfiles):
        print('process:%dth %s '%(i,file))
        img_file=file.replace(label_path,dataset_path).replace('txt','jpg')
        count=ssd_tool.num_boxes
        img_shape, bndboxes, labels=utils.get_imgInfo(img_file)
        img, img_data, bndboxes = utils.process_image(img_file,
                                                      img_shape, bndboxes, (512, 512))
        # ssd_tool.bboxes_Histogram(bndboxes,ssd_anchors,labels)
        # calc_bndboxes(bndboxes,labels,boxes_dict,labels_dict)
        imgname=img_file.split('\\')[-1]
        imgname=imgname.split('.')[0]
        feat_loc,feat_score,feat_labels=ssd_tool.encode_bboxes(ssd_anchors,bndboxes,labels,imgname)

        # ssd_tool.crop_imgage(cv2.imread(imgname),feat_loc,feat_labels,feat_score,imgname)
        count=ssd_tool.num_boxes-count
        # if not count:
        #     i+=1
        #     # print('error:%s %d'%(file,len(bndboxes)))

    result = sorted(ssd_tool.catched_boxes.items(), key=lambda x: x[1], reverse=True)
    print(result)

    catch_array = np.array(list(ssd_tool.catched_boxes.values()))
    print('捕捉到的bboxes:', np.count_nonzero(catch_array))
    print('总共：', np.sum(catch_array))

    print(ssd_tool.bboxes_dict)
    print(ssd_tool.labels_dict)
    total_bboxes=16*16+8*8+4*4
    total_bboxes*=3.*len(txtfiles)
    total_positive=np.sum(list(ssd_tool.bboxes_dict.values()))
    print('total pos num:%d'%total_positive)
    print('positive ratio:%f'%(total_positive/total_bboxes))

    plt.figure(1)
    plt.bar(ssd_tool.bboxes_dict.keys(),ssd_tool.bboxes_dict.values())
    plt.figure(2)
    plt.bar(ssd_tool.labels_dict.keys(),ssd_tool.labels_dict.values())
    # plt.figure(3)
    # plt.bar(labels_dict.keys(),labels_dict.values())
    # plt.figure(4)
    # plt.bar(boxes_dict.keys(),boxes_dict.values())
    plt.show()


