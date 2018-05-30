# -*- coding: utf-8 -*-
import numpy as np
import os,random
import skimage
import sys
import caffe
import sklearn.metrics.pairwise as pw
import math

#  sys.path.insert(0, '/Downloads/caffe-master/python');
#  load Caffe model

caffe.set_mode_gpu()

global net
net = caffe.Classifier('VGG_FACE_deploy.prototxt', 'VGG_FACE.caffemodel')

#def compare_pic(feature1, feature2):
#    predicts = pw.cosine_similarity(feature1, feature2)
#    return predicts


#def get_feature(path):
#    global net
#    X = read_image(path)
#    # test_num = np.shape(X)[0];
#    # print test_num;
#    out = net.forward_all(data=X)
#    feature = np.float64(out['deepid'])
#    feature = np.reshape(feature, (1, 160))
#    return feature

#def read_image(filepath):
#    averageImg = [129.1863, 104.7624, 93.5940]
#    X = np.empty((1, 3, 144, 144))
#    filename = filepath.split('\n')
#    filename = filename[0]
#    im = skimage.io.imread(filename, as_grey=False)
#    image = skimage.transform.resize(im, (144, 144)) * 255
#    #mean_blob.shape = (-1, 1);
#    #mean = np.sum(mean_blob) / len(mean_blob);
#    X[0, 0, :, :] = image[:, :, 0] - averageImg[0]
#    X[0, 1, :, :] = image[:, :, 1] - averageImg[1]
#    X[0, 2, :, :] = image[:, :, 2] - averageImg[2]
#    return X
# 零均值化
def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    newData = dataMat - meanVal
    return newData, meanVal

def pca(dataMat, n):
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
    n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
    lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据
    return lowDDataMat, reconMat

def compar_pic(path1, path2):
    global net
    # 加载验证图片
    X = read_image(path1)
    test_num = np.shape(X)[0]
    # X  作为 模型的输入
    out = net.forward_all(data=X)
    # fc7是模型的输出,也就是特征值
    # feature1 = np.float64(out['fc7'])
    feature1 = np.float64(net.blobs['fc8'].data)
    feature1 = np.reshape(feature1, (test_num, 2622))
    feature1_conv5_3 = np.float64(net.blobs['conv5_3'].data)
    feature1_conv5_3 = np.reshape(feature1_conv5_3, (2, 50176))
    lowDDataMat, reconMat = pca(feature1_conv5_3, 1000)
    feature1_conv5_3 = np.reshape(lowDDataMat, (1, 2000))
    # np.savetxt('feature1.txt', feature1, delimiter=',')
    # 加载注册图片
    X = read_image(path2)
    # X  作为 模型的输入
    out = net.forward_all(data=X)
    # fc7是模型的输出,也就是特征值
    # feature2 = np.float64(out['fc7'])
    feature2 = np.float64(net.blobs['fc8'].data)
    feature2 = np.reshape(feature2, (test_num, 2622))
    feature2_conv5_3 = np.float64(net.blobs['conv5_3'].data)
    feature2_conv5_3 = np.reshape(feature2_conv5_3, (2, 50176))
    lowDDataMat, reconMat = pca(feature2_conv5_3, 1000)
    feature2_conv5_3 = np.reshape(lowDDataMat, (1, 2000))
    # np.savetxt('feature2.txt', feature2, delimiter=',')
    # 求两个特征向量的cos值,并作为是否相似的依据
    predicts = pw.cosine_similarity(feature1_conv5_3, feature2_conv5_3)
    return predicts

def read_image(filelist):
    averageImg = [129.1863, 104.7624, 93.5940]
    X = np.empty((1, 3, 224, 224))
    word = filelist.split('\n')
    filename = word[0]
    im1 = skimage.io.imread(filename, as_grey=False)
    image = skimage.transform.resize(im1, (224, 224)) * 255
    X[0, 0, :, :] = image[:, :, 0] - averageImg[0]
    X[0, 1, :, :] = image[:, :, 1] - averageImg[1]
    X[0, 2, :, :] = image[:, :, 2] - averageImg[2]
    return X

# Iterate file system.
def saveFileInfo(file_list, output_path):
    #print "Writing file info to", output_path
    with open(output_path, mode = 'a') as f:
        for filenames in file_list:
            for item in filenames:
                line = ''.join(str(item)) + '\n'
                f.write(line)
        f.close()

if __name__ == '__main__':
    thershold = 0
    accuracy = 0
    thld = 0
    TEST_SUM = 500
    DATA_BASE = "F:/publicData/YouTubeFaces/aligned_images_DB"
    POSITIVE_TEST_FILE = "./data/data_pairs/split3_positive.txt"
    NEGATIVE_TEST_FILE = "./data/data_pairs/split3_negative.txt"
    POSITIVE_SIMILARIY = "./data/data_pairs_similarity/similarity_split3_positive_fc8.txt"
    NEGATIVE_SIMILARIY = "./data/data_pairs_similarity/similarity_split3_negative_fc8.txt"

    # Positive Test
    f_positive = open(POSITIVE_TEST_FILE, "r")
    PositiveDataList = f_positive.readlines()
    f_positive.close()
    f_negative = open(NEGATIVE_TEST_FILE, "r")
    NegativeDataList = f_negative.readlines()
    f_negative.close()
    for index in range(len(PositiveDataList)):
        filepath_1 = PositiveDataList[index].split(' ')[0]
        filepath_2 = PositiveDataList[index].split(' ')[1][:-1]
        files_1 = os.listdir(DATA_BASE + '/'+  filepath_1)
        samples_1 = random.sample(files_1, 1)
        for sample1 in samples_1:
            sample1 = DATA_BASE + '/' + filepath_1 + '/' + sample1
        files_2 = os.listdir(DATA_BASE + '/' + filepath_2)
        samples_2 = random.sample(files_2, 1)
        for sample2 in samples_2:
            sample2 = DATA_BASE + '/' + filepath_2 + '/' + sample2
        #feature_1 = get_feature(DATA_BASE + filepath_1)
        #feature_2 = get_feature(DATA_BASE + filepath_2)
        print PositiveDataList[index]
        print filepath_2
        result_p = compar_pic(sample1, sample2)
        print result_p
        saveFileInfo(result_p.tolist(),POSITIVE_SIMILARIY)

    for index in range(len(NegativeDataList)):
        filepath_1 = NegativeDataList[index].split(' ')[0]
        filepath_2 = NegativeDataList[index].split(' ')[1][:-1]
        #feature_1 = get_feature(DATA_BASE + filepath_1)
        #feature_2 = get_feature(DATA_BASE + filepath_2)
        files_1 = os.listdir(DATA_BASE + '/' + filepath_1)
        samples_1 = random.sample(files_1, 1)
        for sample1 in samples_1:
            sample1 = DATA_BASE + '/' + filepath_1 + '/' + sample1
        files_2 = os.listdir(DATA_BASE + '/' + filepath_2)
        samples_2 = random.sample(files_2, 1)
        for sample2 in samples_2:
            sample2 = DATA_BASE + '/' + filepath_2 + '/' + sample2
        result_n = compar_pic(sample1, sample2)
        print result_n
        saveFileInfo(result_n.tolist(),NEGATIVE_SIMILARIY)

    for thershold in np.arange(0.10, 0.99, 0.01):
        True_Positive = 0
        True_Negative = 0
        False_Positive = 0
        False_Negative = 0
        fp = open(POSITIVE_SIMILARIY, 'r')
        lines1 = fp.readlines()
        for line1 in lines1:
            if float(line1) >= thershold:
                #  print 'Same Guy\n\n'
                True_Positive += 1
            else:
                #  wrong
                False_Positive += 1
        fn = open(NEGATIVE_SIMILARIY, 'r')
        lines2 = fn.readlines()
        for line2 in lines2:
            if float(line2) >= thershold:
                #  print 'Wrong Guy\n\n'
                #  wrong
                False_Negative += 1
            else:
                #  correct
                True_Negative += 1

        print "thershold: " + str(thershold)
        print "Accuracy: " + str(float(True_Positive + True_Negative) / TEST_SUM) + " %"
        print "True_Positive: " + str(float(True_Positive) / TEST_SUM) + " %"
        print "True_Negative: " + str(float(True_Negative) / TEST_SUM) + " %"
        print "False_Positive: " + str(float(False_Positive) / TEST_SUM) + " %"
        print "False_Negative: " + str(float(False_Negative) / TEST_SUM) + " %"

        if accuracy < float(True_Positive + True_Negative) / TEST_SUM:
            accuracy = float(True_Positive + True_Negative) / TEST_SUM
            thld = thershold
    print 'Best performance: %f, with threshold %f ' % (accuracy, thld)
