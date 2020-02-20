import numpy as np
import cv2
import glob
import os
import imageio
from tqdm import tqdm
import tensorflow as tf


dataset_path = [r'I:\Code\TailMineDetection\dataset\positiveSamples20191118']
# dataset_path.append(os.path.abspath(os.path.join(dataset_path[-1], '..', 'val')))
imgType = '.jpg'


def main(argv):
    print('create tfrecords files')
    for phase in dataset_path:
        img_files = glob.glob(os.path.join(phase, r'*%s' % imgType))
        resultPath='../Tfrecords/Tail/' + phase.split('\\')[-1]
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)
        pbar = tqdm(total=len(img_files))
        with tf.Session() as sess:
            with tf.python_io.TFRecordWriter('%s/LEVIR_%s.tfrecord' % (resultPath, phase.split('\\')[-1])) as tf_writer:
                for index, img_file in enumerate(img_files):
                    pbar.update(1)
                    img_shape, bndboxes, labels = get_img_info(img_file)
                    img_data, img, bndboxes = process_img(img_file, img_shape, bndboxes, (512,512))
                    # mask = get_mask(img_shape, bndboxes)
                    # 测试是否有越界数据
                    if len(labels):
                        for box in bndboxes:
                            xmin, ymin, xmax, ymax = box
                            if xmin<0 or xmin>1 or ymin<0 or ymin>1  or xmax<0 or xmax>1 or ymax<0 or ymax>1:
                                print(box)
                                print(img_file)

                    if len(labels)==0 or labels[0] == 0:
                        for _ in range(1):
                            examples = create_tfrecords(img_data, bndboxes, labels, img_file)
                            tf_writer.write(examples.SerializeToString())
                    else:
                        for _ in range(3):
                            examples = create_tfrecords(img_data, bndboxes, labels, img_file)
                            tf_writer.write(examples.SerializeToString())
        print('finished save %s tfrecords' % (phase.split('\\')[-1]))
        pbar.close()

def get_mask(img_shape, bboxes):
    bboxes = np.trunc(np.array(bboxes, np.float)*7.99).astype(int)
    mask = np.zeros([8,8],np.int32)
    for box in bboxes:
        mask[box[1],box[0]] = 1
        mask[box[1],box[2]] = 1
        mask[box[3],box[0]] = 1
        mask[box[3],box[2]] = 1
    mask=np.reshape(mask,-1)

    return [x for x in mask]

def get_img_info(img_file):
    txt_file = img_file.replace(imgType, '.txt')
    img = imageio.imread(img_file)
    img_height, img_width, img_channel = img.shape
    labels = np.array([0])
    bndboxes = np.array([[0,0,0,0]])
    if not os.path.exists(txt_file):
        print('continue')
        return (img_height, img_width), bndboxes, labels
    if os.path.getsize(txt_file) > 0:
        data = np.loadtxt(txt_file, skiprows=0, dtype=np.int, ndmin=2)
        if data.size > 0:
            labels=data[:, 0]
            bndboxes=data[:, 1:5]
            # 其中几类
            # bndboxes=bndboxes[[x in [1, 2, 3] for x in labels]]
            # labels = labels[[x in [1, 2, 3] for x in labels]]
            # bndboxes=bndboxes[labels==1]
            # labels=labels[labels==3]
            # labels[labels==3]=1

    labels=labels.tolist()
    return (img_height, img_width), bndboxes, labels


def process_img(img_file, img_shape, bndboxes, out_shape):
    img_data = tf.gfile.FastGFile(img_file,'rb').read()
    img = tf.image.decode_jpeg(img_data)
    img=tf.image.convert_image_dtype(img, tf.float32)
    img=tf.image.resize_images(img, out_shape)
    out_bndboxes=[]
    height_ratios = out_shape[0]/img_shape[0]
    width_ratios = out_shape[1]/img_shape[1]
    for box in bndboxes:
        ymin = box[1] * height_ratios / out_shape[0]
        xmin = box[0] * width_ratios / out_shape[1]
        ymax = box[3] * height_ratios / out_shape[0]
        xmax = box[2] * width_ratios / out_shape[1]
        out_bndboxes.append((max(float(ymin), 0.0), max(float(xmin), 0.0), min(float(ymax), 1.0), min(float(xmax), 1.0)))
    return img_data, img, out_bndboxes

def create_tfrecords(img, bndboxes, labels, img_name,img_shape=[512,512],img_channel=3):
    ymin=[]
    xmin=[]
    ymax=[]
    xmax=[]
    for b in bndboxes:
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]

    features = tf.train.Example(features=tf.train.Features(feature={
        'img': bytes_feature(img),
        'img/format': bytes_feature('jpg'.encode('utf-8')),
        'img/height': float_feature(img_shape[0]),
        'img/width': float_feature(img_shape[1]),
        'img/channel': int64_feature(img_channel),
        'img/name': bytes_feature(bytes(img_name,encoding='utf-8')),
        'bndboxes/ymin': float_feature(ymin),
        'bndboxes/xmin': float_feature(xmin),
        'bndboxes/ymax': float_feature(ymax),
        'bndboxes/xmax': float_feature(xmax),
        'labels': int64_feature(labels),
        'difficults': int64_feature(0),
        'labels_num': int64_feature(len(labels)),
    }))
    return features

def bytes_feature(value):
    if not isinstance(value, list):
        value= [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def draw_rectangle(imgFile,index,bndboxes,labels):
    img=cv2.imread(imgFile)
    print(imgFile)
    img=cv2.resize(img,(512,512))
    for i in  range(len(bndboxes)):
        boxes=bndboxes[i]
        label=labels[i]
        ymin=int(boxes[0]*512)
        xmin=int(boxes[1]*512)
        ymax=int(boxes[2]*512)
        xmax=int(boxes[3]*512)
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),1)
        cv2.putText(img,str(label),(xmin+20,ymin+10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
    # img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imwrite('testResultImg/'+str(index)+'_'+imgFile.split('\\')[-1],img)

if __name__=='__main__':
    tf.app.run()