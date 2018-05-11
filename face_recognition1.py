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
def recog(input,dataset,namelist):
    global face_rect
    #src_path='./regist_pic/'+str(md)
    #src_path = input.split(' ').[0]
    data = dataset
    #label = namelist
    ret = 0
    m = 0
    acc = 0
    all_people = 1595
    f1 = file(input, 'r')
    f1_list = f1.readlines()
    f2 = open(namelist,'r')
    f2_list = f2.readlines()

    for index1 in xrange(len(f1_list)):
        global ret
        global ID
        img = f1_list[index1].split(' ')[0]
        label1 = f1_list[index1].split(' ')[1][:-1]
        print '正在测试第' + str(index1) + '张人脸'
        for index2 in xrange(len(f2_list)):
            #img = f1_list[index1].split(' ')[0]
            #label1 = f1_list[index1].split(' ')[1][:-1]
            line = f2_list[index2].split(' ')[0]
            label2 = f2_list[index2].split(' ')[1][:-1]
            #print label2
            datalist = data + '/' + line
            imglist = data + '/' + img
            res = compar_pic(datalist, imglist)
            #result = scores.append(res)
            if ret < res:
                ret = res
                ID = label2
            m += 1
            #print str(res) + ' ' + str(ret)
        if ID == label1:
            acc += 1
        else:
            print '预测不正确的ID是:' + str(label1) + '，预测为: ' + str(ID)
        print str(label1) + '的预测得分是：' + str(ret)
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
    namelist = './data/images_4.txt'
    dataset = 'F:/dataset/YouTubeFaces/frame_images_DB'
    input = './data/part2-5.txt'
    ret = 0
    recog(input,dataset,namelist)
