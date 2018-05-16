# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
#import cv2.cv as cv
import cv2 as cv
#from cv2 import cv as cv
from skimage import transform as tf
from PIL import Image, ImageDraw
import threading
from time import ctime,sleep
import time
import sklearn
import matplotlib.pyplot as plt
import skimage

#caffe_root = '/home/gk/caffe-master/'
#import sys
#sys.path.insert(0, caffe_root + 'python')
import caffe

import sklearn.metrics.pairwise as pw

#保存人脸的位置
global face_rect
face_rect=[]
#caffe.set_mode_gpu()

#加载caffe模型
global net
net=caffe.Classifier('./VGG_FACE_deploy.prototxt','./VGG_FACE.caffemodel')
#用来识别一个用户
def visualization(data,head,padsize = 1, padval = 0):
    #data -= data.min()
    #print data
    #data += data.max()
    #print data
    data = (data - data.min()) / (data.max() - data.min())
    # 强制性地使输入的图像个数为平方数，不足平方数时，手动添加几幅
    n = int(np.ceil(np.sqrt(data.shape[0])))
    print "n = ", n
    #强制滤波器/卷积核数量为偶数 # 每幅小图像之间加入小空隙
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    #给卷积核命名 # 将所有输入的data图像平复在一个ndarray-data中（tile the filters into an image）
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure()
    plt.title(head)
    plt.imshow(data)
    plt.axis('off')
    plt.show()
def recog(test_input,dataset,namelist):
    global face_rect
    #src_path='./regist_pic/'+str(md)
    #src_path = input.split(' ').[0]
    data = dataset
    #label = namelist
    ret = 0
    m = 0
    acc = 0
    all_people = 1004
    f1 = file(test_input, 'r')
    f1_list = f1.readlines()
    f2 = open(namelist,'r')
    f2_list = f2.readlines()

    for index1 in xrange(len(f1_list)):
        global ret
        global ID
        img_test = f1_list[index1].split(' ')[0]
        label_test = f1_list[index1].split(' ')[1][:-1]
        print '正在测试第' + str(index1) + '张人脸'
        for index2 in xrange(len(f2_list)):
            #img = f1_list[index1].split(' ')[0]
            #label1 = f1_list[index1].split(' ')[1][:-1]
            img_register = f2_list[index2].split(' ')[0]
            label_register = f2_list[index2].split(' ')[1][:-1]
            #print label2
            datalist = data + '/' + img_register
            imglist = data + '/' + img_test
            res = compar_pic(datalist, imglist)
            #result = scores.append(res)
            if ret < res:
                ret = res
                ID = label_register
            m += 1
            #print str(res) + ' ' + str(ret)
        if ID == label_test:
            acc += 1
        else:
            print '预测不正确的ID是:' + str(label_test) + '，预测为: ' + str(ID)
        print str(label_test) + '的预测得分是：' + str(ret)
        ret = 0
        ID = 5
        print '当前acc = ' + str(acc)
    acc = float(acc/all_people)
    #print '识别成功!!!!\n' + '他的ID是:' + str(label2)
    print '识别率为：' + str(acc)
    #print m
    #return result
def compar_pic(path1,path2):
    global net
    #加载验证图片
    X=read_image(path1)
    test_num=np.shape(X)[0]
    #X  作为 模型的输入
    out = net.forward_all(data = X)
    #fc7是模型的输出,也就是特征值
    #visualization(net.blobs['conv1_1'].data[0], 'post-conv1_1 images')
    #visualization(net.params['conv1_1'][0].data[:,0], 'conv1_1 weights(filter)')
    #visualization(net.blobs['conv1_2'].data[0], 'post-conv1_2 images')
    #visualization(net.blobs['pool1'].data[0], 'post-pool1 images')
    #visualization(net.blobs['conv2_1'].data[0], 'post-conv2_1 images')
    #visualization(net.blobs['conv2_2'].data[0], 'post-conv2_2 images')
    #visualization(net.blobs['pool2'].data[0], 'post-pool2 images')
    #visualization(net.blobs['conv3_1'].data[0], 'post-conv3_1 images')
    #visualization(net.blobs['conv3_2'].data[0], 'post-conv3_2 images')
    #visualization(net.blobs['conv3_3'].data[0], 'post-conv3_3 images')
    #visualization(net.blobs['pool3'].data[0], 'post-pool3 images')
    #visualization(net.blobs['conv4_1'].data[0], 'post-conv4_1 images')
    #visualization(net.blobs['conv4_2'].data[0], 'post-conv4_2 images')
    #visualization(net.blobs['conv4_3'].data[0], 'post-conv4_3 images')
    #visualization(net.blobs['pool4'].data[0], 'post-pool4 images')
    #visualization(net.blobs['conv5_1'].data[0], 'post-conv5_1 images')
    #visualization(net.blobs['conv5_2'].data[0], 'post-conv5_2 images')
    #visualization(net.blobs['conv5_3'].data[0], 'post-conv5_3 images')
    #visualization(net.blobs['pool5'].data[0], 'post-pool5 images')
    #net.params[‘layername’][0].data
    #visualization(net.params['conv1_1'][0].data[:,0], 'conv weights(filter)')
    #visualization(net.params['conv1_2'][0].data[:, 0], 'conv weights(filter)')
    #visualization(net.params['conv1_3'][0].data[:, 0], 'conv weights(filter)')
    feat = net.blobs['prob'].data[0]
    plt.figure(figsize=(15, 3))
    plt.plot(feat.flat)
    feat1 = net.blobs['fc7'].data[0]
    plt.subplot(2, 1, 1)
    plt.plot(feat1.flat)
    plt.subplot(2, 1, 2)
    plt.hist(feat1.flat[feat1.flat > 0], bins=100)
    feature1 = np.float64(net.blobs['fc7'].data)
    feature1 = np.reshape(feature1,(test_num,4096))
    #np.savetxt('feature1.txt', feature1, delimiter=',')

    #加载注册图片
    X=read_image(path2)
    #X  作为 模型的输入
    out = net.forward_all(data=X)
    #fc7是模型的输出,也就是特征值
    feat2 = net.blobs['fc7'].data[0]
    plt.subplot(2, 1, 1)
    plt.plot(feat2.flat)
    plt.subplot(2, 1, 2)
    plt.hist(feat2.flat[feat2.flat > 0], bins=100)
    feature2 = np.float64(net.blobs['fc7'].data)
    feature2=np.reshape(feature2,(test_num,4096))
    #np.savetxt('feature2.txt', feature2, delimiter=',')
    #求两个特征向量的cos值,并作为是否相似的依据
    predicts=pw.cosine_similarity(feature1, feature2)
    return  predicts

def read_image(filelist):
    averageImg = [129.1863,104.7624,93.5940]
    X=np.empty((1,3,224,224))
    word=filelist.split('\n')
    filename=word[0]
    #filename = filelist
    im1=skimage.io.imread(filename,as_grey=False)
    image =skimage.transform.resize(im1,(224, 224))*255
    X[0,0,:,:]=image[:,:,0]-averageImg[0]
    X[0,1,:,:]=image[:,:,1]-averageImg[1]
    X[0,2,:,:]=image[:,:,2]-averageImg[2]
    return X

if __name__ == '__main__':
    namelist = './data/register1_aligned.txt'
    dataset = 'F:/dataset/YouTubeFaces/aligned_images_DB'
    test_input = './data/test_aligned.txt'
    ret = 0
    recog(test_input,dataset,namelist)
