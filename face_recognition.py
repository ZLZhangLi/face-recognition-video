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

#caffe_root = '/home/zl/caffe-master/'
#import sys
#sys.path.insert(0, caffe_root + 'python')
import caffe

import sklearn.metrics.pairwise as pw

#保存人脸的位置
global face_rect
face_rect=[]
caffe.set_mode_gpu()

#加载caffe模型
global net
net=caffe.Classifier('./VGG_FACE_deploy.prototxt','./VGG_FACE.caffemodel')
#用来识别一个用户
def recog(input,dataset,namelist):
    global face_rect
    #src_path='./regist_pic/'+str(md)
    src_path = input
    data = dataset
    label = namelist
    ret = 0
    m = 0
    f = file(label,'r')
    for line in f.readlines():
        datalist = data + '/' + line
        res = compar_pic(datalist, src_path)
        #result = scores.append(res)
        if ret < res:
            ret = res
            ID = datalist
        m += 1
        print str(res) + ' ' + str(ret)
    print '识别成功!!!!\n' + '他的ID是:' + str(ID)
    print m
    #return result
def compar_pic(path1,path2):
    global net
    #加载验证图片
    X=read_image(path1)
    test_num=np.shape(X)[0]
    #X  作为 模型的输入
    out = net.forward_all(data = X)
    #fc7是模型的输出,也就是特征值
    #print out
    print net
    print net.blobs['fc7']
    print net.blobs['fc7'].data
    net
    net.blobs['fc7']
    net.blobs['fc7'].data
    feature1 = np.float64(net.blobs['fc7'].data)
    feature1 = np.reshape(feature1,(test_num,4096))
    #np.savetxt('feature1.txt', feature1, delimiter=',')

    #加载注册图片
    X=read_image(path2)
    #X  作为 模型的输入
    out = net.forward_all(data=X)
    #fc7是模型的输出,也就是特征值
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
    namelist = './data/sample_same_frame_images.txt'
    dataset = 'F:/publicData/YouTubeFaces/frame_images_DB'
    input = 'F:/publicData/YouTubeFaces/frame_images_DB/Ahmed_Chalabi/2/2.521.jpg'
    #while True:
    ret = 0
    recog(input,dataset,namelist)
