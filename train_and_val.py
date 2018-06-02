#! /usr/bin/env python
#-*- coding:utf-8 -*-
import os,random
# Iterate file system.
def loadInfo(dataset_root):
	print "Searching files in", dataset_root
	folders = sorted(os.listdir(dataset_root))
	train_info = []
	val_info = []
	label = 1
	num = 1
	for folder in folders:
		folder_path = dataset_root + '/' + folder
		items = os.listdir(folder_path)
		for item in range(len(items)):
			item_path = folder_path + '/' + items[item]
			if not os.path.isdir(item_path):
				continue
			files = os.listdir(item_path)
			for file in files:
				if num%10!=0:
					train_path = folder + '/' + items[item] + '/' + str(file) + ' ' +  str(label)
					train_info.append(train_path)
				else:
					val_path = folder + '/' + items[item] + '/' + str(file) + ' ' + str(label)
					val_info.append(val_path)
				num += 1
		label += 1
	return train_info, val_info

def saveFileInfo(file_list, output_path):
	print "Writing file info to", output_path
	with open(output_path, 'w') as f:
		#label = 0
		for items in file_list:
			#for item in items:
			#line = ' '.join([items,str(label)]) + '\n'
			line = ''.join(items) + '\n'
			#line = ' '.join([item[item.find('lfw-cropped')+12:],str(label)]) + '\n'
			f.write(line)
		#	label += 1
		f.close()
if __name__ == '__main__':
	root_path = 'F:/publicData/YouTubeFaces/aligned_images_DB2'
	train_info, val_info = loadInfo(root_path)
	train_path = './data/train_val_txt/train.txt'
	val_path = './data/train_val_txt/val.txt'
	print 'save train.txt...'
	saveFileInfo(train_info, train_path)
	print 'save val.txt...'
	saveFileInfo(val_info, val_path)
