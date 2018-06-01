#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import caffe
TOOLS = 'F:/zhangli_code/caffe-master/caffe-master/Build/x64/Release/'
def convert_lmdb(Image_Root, file_image_list,lmdb_path, save_image_size=(224,224)):
    command =TOOLS+'convert_imageset --resize_height='+str(save_image_size[0])+' --resize_width='+str(save_image_size[1])  +'  --shuffle '+Image_Root+' ' +file_image_list+' '+lmdb_path
    print command
    os.system(command)
    print 'convert done!'

def caculate_image_mean(lmdb_path, save_path):
    command_mean =TOOLS+'compute_image_mean '+  lmdb_path+' '+save_path
    print command_mean
    os.system(command_mean)
    print 'caculate image mean done!'

if __name__ == "__main__":
    Image_Root = 'F:/publicData/YouTubeFaces/aligned_images_DB2/'
    label_train_txt = './data/train_val_txt/train.txt'
    label_test_txt = './data/train_val_txt/val.txt'
    lmdb_train_path = 'I:/ytf_lmdb/train_lmdb/'
    lmdb_test_path = 'I:/ytf_lmdb/val_lmdb/'
    save_train_mean_path = 'I:/ytf_lmdb/meanfile/mean_train.binaryproto'
    save_test_mean_path = 'I:/ytf_lmdb/meanfile/mean_val.binaryproto'
    save_image_size = (224,224)
    #convert_lmdb(Image_Root,label_train_txt,lmdb_train_path,save_image_size);
    #print 'train convert done!'
    convert_lmdb(Image_Root,label_test_txt,lmdb_test_path,save_image_size);
    print 'test convert done!'
    caculate_image_mean(lmdb_train_path,save_train_mean_path)
    print 'caculate train image mean done!'
    caculate_image_mean(lmdb_test_path,save_test_mean_path)
    print 'caculate test image mean done!'