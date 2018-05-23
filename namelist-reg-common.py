#! /usr/bin/env python
#-*- coding:utf-8 -*-
import os,random
# Iterate file system.
def loadInfo(dataset_root):
	print "Searching files in", dataset_root
	folders = sorted(os.listdir(dataset_root))
	file_info = []
	label = 1
	for folder in folders:
		folder_path = dataset_root + '/' + folder
		items = os.listdir(folder_path)
		for item in range(len(items)):
			#if len(items) < 2:
			#	break
			#m = item + 1
			#if m > len(items)-1:
			#	break
			#item_path = folder_path + '/' + items[item]
			item_path = folder_path + '/' + items[item]
			if not os.path.isdir(item_path):
				continue
			files = os.listdir(item_path)
			#每个视频中取出1帧
			samples= random.sample(files, 4)
			for sample in samples:
				#sample_path = folder + '/' + items[item] + '/' + str(sample)[2:-2] + ' ' + str(label)
				sample_path = folder + '/' + items[item] + '/' + str(sample) + ' ' + str(label)
				file_info.append(sample_path)
			#file_info.append(map(lambda x: folder + '/' + items[item] + '/' + x, files))
			#每个帧中取一张图片
			#for file in files:
			#	file = folder + '/' + items[item] + '/' + file + ' ' + str(label)
			#	file_info.append(file)
			#	break
			#label += 1
			#break
		label += 1
	return file_info

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
	root_path = 'F:/publicData/YouTubeFaces/aligned_images_DB'
	file_info = loadInfo(root_path)
	output_path = './data/aligned_register_4s.txt'
	saveFileInfo(file_info, output_path)
