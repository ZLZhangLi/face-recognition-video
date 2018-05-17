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
import skimage, pickle

#caffe_root = '/home/gk/caffe-master/'
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
#net=caffe.Classifier('./ResNet-50-deploy.prototxt','./ResNet-50-model.caffemodel')
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
# Save feature and labels to file.
def saveData(feat, labels, filename):
	print('Saving data to %s...' % (filename))
	with open(filename, 'wb') as f:
		pickle.dump(feat, f)
		pickle.dump(labels, f)
		f.close()
	pass

# Load feature and labels from file.
def loadData(filename):
	print('Loading data to %s...' % (filename))
	with open(filename, 'rb') as f:
		feat = pickle.load(f)
		labels = pickle.load(f)
		f.close()
	return (feat, labels)
#用来识别一个用户
def register(data,namelist,out_feature):
    feat_all = np.array(None)
    label_list = []

    f = open(namelist, 'r')
    f_list = f.readlines()
    for index in xrange(len(f_list)):
        img_register = f_list[index].split(' ')[0]
        label_register = f_list[index].split(' ')[1][:-1]
        #label = float(label_register)
        #n_label = np.asarray(label_register,'f')
        #n_label = np.array(label_register,dtype = 'float_')
        datalist = data + '/' + img_register
        feature = get_feat(datalist)
        if feat_all.any():  # .any()方法 如果有一个不为零，则为True 反之为False
            feat_all = np.concatenate((feat_all, feature))  # 将feat_all和deepid拼接在一起
        else:
            feat_all = feature
        if len(label_list):
            #label_list = label_register
            label_list.extend(list(label_register))
        else:
            #label_list = label_list.extend(label_register)
            label_list = list(label_register)
    label_all = np.array(label_list)
    return feat_all, label_all

'''
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
            get_feat(datalist)
            #res = compar_pic(datalist, imglist)
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
    '''
def get_feat(path):
    global net
    # 加载验证图片
    X = read_image(path)
    test_num = np.shape(X)[0]
    # X  作为 模型的输入
    out1 = net.forward_all(data=X)
    # feat = net.blobs['prob'].data[0]
    # plt.figure(figsize=(15, 3))
    # plt.plot(feat.flat)
    # fc7是模型的输出,也就是特征值
    # feat1 = net.blobs['fc7'].data[0]
    # plt.subplot(2, 1, 1)
    # plt.plot(feat1.flat)
    # plt.subplot(2, 1, 2)
    # plt.hist(feat1.flat[feat1.flat > 0], bins=100)
    # visualization(net.blobs['res5b_branch2a'].data[0], 'post-conv1_1 images')
    # visualization(net.blobs['bn_conv1'].data[0], 'post-conv1_1 images')
    # visualization(net.blobs['res5b_branch2c'].data[0], 'post-conv1_1 images')
    # visualization(net.blobs['pool5'].data[0], 'post-conv1_1 images')
    feature = np.float64(net.blobs['fc7'].data)
    # feature1 = np.float64(out1['prob'])
    feature = np.reshape(feature, (test_num, 4096))
    return feature

def compar_pic(features_1, features_2, labels_1, labels_2):

    #np.savetxt('feature1.txt', feature1, delimiter=',')

    #加载注册图片
    #Y=read_image(path2)
    #Y  作为 模型的输入
    #out2 = net.forward_all(data=Y)
    #feat2 = net.blobs['fc_ft'].data[0]
    #plt.subplot(2, 1, 1)
    #plt.plot(feat2.flat)
    #plt.subplot(2, 1, 2)
    #plt.hist(feat2.flat[feat2.flat > 0], bins=100)
    #fc7是模型的输出,也就是特征值
    #feature2 = np.float64(net.blobs['fc1000'].data)
    #feature2 = np.float64(out2['prob'])
    #feature2=np.reshape(feature2,(test_num,1000))
    #np.savetxt('feature2.txt', feature2, delimiter=',')
    #求两个特征向量的cos值,并作为是否相似的依据
    res = 0
    acc = 0
    #bool index1
    for i in xrange(features_1.shape[0]):
        #index1 = np.argwhere(features_1 == feature1)
        f1 = features_1[i]
        feature1 = np.array([f1])
        label_1 = labels_1[i]
        #for label_1 in labels_1:
        for j in xrange(features_2.shape[0]):
                #for label_2 in labels_2:
            f2 = features_2[j]
            feature2 = np.array([f2])
            label_2 = labels_2[j]
            similarity = pw.cosine_similarity(feature1, feature2)
            if res < similarity:
                id = label_2
                res = similarity
        if label_1 == id:
            acc += 1
        else:
            print '预测不正确的ID是:' + str(label_1) + '，预测为: ' + str(id)
    print str(label_1) + '的预测得分是：' + str(res)
    print '当前acc = ' + str(acc)
    #acc = float(acc / all_people)
    # print '识别成功!!!!\n' + '他的ID是:' + str(label2)
    print '识别率为：' + str(acc)

    #predicts=pw.cosine_similarity(path1, path2)
    #return  predicts

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
    dataset = 'F:/publicData/YouTubeFaces/frame_images_DB'
    test_input = './data/images_onlyone.txt'
    register_feature = './data/register_feature.pkl'
    test_feature = './data/test_feature.pkl'
    ret = 0
    print '正在提取注册数据的特征和标签！！！'
    r_feature,r_label = register(dataset, namelist, register_feature)
    print '提取完毕！！！'
    saveData(r_feature, r_label, register_feature)
    print '正在提取识别数据的特征和标签！！！'
    t_feature, t_label = register(dataset, test_input, test_feature)
    print '提取完毕！！！'
    saveData(t_feature, t_label, test_feature)
    load_r_features, load_r_labels = loadData(register_feature)
    load_t_features, load_t_labels = loadData(test_feature)
    compar_pic(load_t_features, load_r_features, load_t_labels, load_r_labels)
