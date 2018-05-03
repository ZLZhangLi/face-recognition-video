#! /usr/bin/env python
#-*- coding:utf-8 -*-
import os
# Iterate file system.
def loadInfo(dataset_root):
	print "Searching files in", dataset_root
	folders = sorted(os.listdir(dataset_root))
	file_info = []
	for folder in folders:
		folder_path = dataset_root + '/' + folder
		if not os.path.isdir(folder_path):
			continue
		#files = sorted(os.listdir(folder_path))
		file_info.append(folder_path)
	return file_info

def saveFileInfo(file_list, output_path):
	print "Writing file info to", output_path
	with open(output_path, 'w') as f:
		label = 0
		for item in file_list:
			#for item in filenames:
			line = ''.join(item[item.find('aligned_images_DB')+18:]) + '\n'
			#line = ' '.join([item[item.find('webface-cropped')+16:],str(label)]) + '\n'
			#line = ' '.join([item[item.find('lfw-cropped')+12:],str(label)]) + '\n'
			f.write(line)
			#label += 1
		f.close()
if __name__ == '__main__':
	#root_path = 'F:/zhangli_code/lfw/lfw-cropped'
	root_path = 'F:/publicData/YouTubeFaces/aligned_images_DB'
	file_info = loadInfo(root_path)
	print file_info
	#output_path = 'F:/zhangli_code/deepid2/DeepID2-master/test/lfw_cropped.txt'
	output_path = 'F:/zhangli_code/face-recognition-video/YTF-label.txt'
	saveFileInfo(file_info, output_path)

