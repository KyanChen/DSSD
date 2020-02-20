import tensorflow as tf
import numpy as np
import cv2
import os
import shutil
import PIL.Image as image
import tqdm
import gdal

from config import FLAGS
from nets.ssd_vgg_512_Multi import SSDNet_vgg_512 as SSDNet
import dataset.dataSet_utils as data_utils

slim = tf.contrib.slim
imgType = '.tiff'

testPath = r'I:\Code\多源遥感大数据目标检测\Predict\Img'
testResultTxtPath = [r'Predict\Img\Result\withoutRAM\Txt', r'Predict\Img\Result\withRAM\Txt']
testReaultImgPath = [r'Predict\Img\Result\withoutRAM\Img', r'Predict\Img\Result\withRAM\Img']
model_path = [r'model\withoutRAM', r'model\withRAM']

scalar = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

def draw_boxes1(img, localizations, scores, labels, img_name=None, threshold=0.5):
    index = np.nonzero(np.logical_and(scores > threshold, labels > 0))
    localizations = localizations * 512
    img = img * 255
    img = np.uint8(img)
    localizations = np.int32(localizations)
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    index = np.array(index)
    index = np.reshape(index, [-1])
    count = 0
    for i in range(len(index)):
        idx = index[i]
        loc = localizations[idx, :]
        # if labels[idx]==1:
        # 	count+=1
        # 	red_color=int(scores[idx]*255)
        # 	cv2.rectangle(img, (loc[1], loc[0]), (loc[3], loc[2]), (red_color, 0, 0),1)
        # 	# cv2.putText(img,str(labels[idx])+'(%.2f)'%scores[idx],(loc[1]+20,loc[0]+10),cv2.FONT_HERSHEY_COMPLEX,.5,(0,255,0))
        # 	mask[loc[0]:loc[2], loc[1]:loc[3]] = 255
        count += 1
        scalar = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (200, 5, 200), (23, 56, 78)]
        cv2.rectangle(img, (loc[1], loc[0]), (loc[3], loc[2]), scalar[labels[idx] - 1], 2)
        # cv2.putText(img,str(labels[idx])+'(%.2f)'%scores[idx],(loc[1]+20,loc[0]+10),cv2.FONT_HERSHEY_COMPLEX, .5,(0,255,0))
        mask[loc[0]:loc[2], loc[1]:loc[3]] = 255
    return img, count, mask

def draw_boxes(img, localizations, scores, labels, img_name, threshold=0.5):
    index = np.nonzero(np.logical_and(scores > threshold, labels > 0))
    localizations = localizations * 512
    img = img * 255
    img = np.uint8(img)
    cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
    localizations = np.int32(localizations)
    index=np.array(index)
    index=np.reshape(index,[-1])
    for i in range(len(index)):
        idx = index[i]
        loc = localizations[idx,:]
        red_color=int(scores[idx]*255)

        cv2.rectangle(img, (loc[1], loc[0]), (loc[3], loc[2]), scalar[labels[idx] - 1], 2)
        cv2.rectangle(img, (loc[1] - 1, loc[0]), (loc[1] + 60, loc[0] - 18), scalar[labels[idx] - 1], cv2.FILLED)
        cv2.putText(img, str(labels[idx]) + ':%.2f' % scores[idx], (loc[1] + 2, loc[0] - 3), cv2.FONT_HERSHEY_COMPLEX,
                    .5, (255, 255, 255))
    img = cv2.resize(img, (1000, 1000))
    cv2.imwrite(img_name, img)
for k in range(0, 2):
    if os.path.exists(testResultTxtPath[k]):
        shutil.rmtree(testResultTxtPath[k])
    os.makedirs(testResultTxtPath[k])
    if os.path.exists(testReaultImgPath[k]):
        shutil.rmtree(testReaultImgPath[k])
    os.makedirs(testReaultImgPath[k])
    with tf.Graph().as_default():
        ssd_nets = SSDNet()
        ssd_anchors = ssd_nets.place_anchors()
        test_img_placeholder = tf.placeholder(tf.float32, [1, None, None, 3])
        test_img = data_utils.preprocess_test(test_img_placeholder)
        test_op = ssd_nets.test_op(test_img, ssd_anchors, reuse=False)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if 'checkpoint' in os.listdir(model_path[k]):
            print('restore from model....')
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(model_path[k])
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

        imgFileList = os.listdir(testPath)
        pbar = tqdm.tqdm(total=len(imgFileList))
        for i, imgFile in enumerate(imgFileList):
            pbar.update(1)
            img_file_name = os.path.join(testPath, imgFile)
            gdal.AllRegister()
            dataset = gdal.Open(img_file_name)
            adfGeoTransform = dataset.GetGeoTransform()

            img = image.open(img_file_name)

            resize_image = img.resize([512, 512], image.BICUBIC)
            part_img = np.expand_dims(resize_image, 0)
            part_img = part_img.astype(np.float32) / 255.
            [img, localization, scores, labels, index, _] = sess.run(test_op, feed_dict={test_img_placeholder: part_img})
            draw_boxes(img, localization[index, :], scores[index], labels[index],
                       os.path.join(testReaultImgPath[k], '%s' % imgFile), 0.5)
            img, localization, scores, labels = img, np.int32(512 * localization[index, :]), scores[index], labels[index]

            txtFile = os.path.join(testResultTxtPath[k], imgFile.replace(imgType, '.txt'))
            with open(txtFile, 'w') as fileToWrite:
                # <class_name> <confidence> <left> <top> <right> <bottom>
                for j in range(0, len(labels)):
                    if labels[j] == 0:
                        continue
                    left, top, right, bottom = localization[j][1] * 1000/512,\
                                               localization[j][0] * 1000/512,\
                                               localization[j][3] * 1000/512,\
                                               localization[j][2] * 1000/512

                    centerX, centerY = (left+right)/2, (top+bottom)/2
                    left = adfGeoTransform[0] + left * abs(adfGeoTransform[1])
                    top = adfGeoTransform[3] + top * abs(adfGeoTransform[5])
                    right = adfGeoTransform[0] + right * abs(adfGeoTransform[1])
                    bottom = adfGeoTransform[3] + bottom * abs(adfGeoTransform[5])

                    data = '%d %.2f %.4f %.4f %.4f %.4f\n' % (labels[j], scores[j], left, top, right, bottom)

                    # if labels[j] == 0:
                    #     continue
                    # for k in range(1, 4):
                    #     data = str(k) + ' ' + str(scores[j, k]) + ' ' + str(localization[j][1]) + ' ' \
                    #            + str(localization[j][0]) + ' ' + str(localization[j][3]) + ' ' + str(localization[j][2]) + '\n'
                    fileToWrite.writelines(data)
            # print(txtFile)
            # gdal.GDALDestroyDriverManager()
        pbar.close()