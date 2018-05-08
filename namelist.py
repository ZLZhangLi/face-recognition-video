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
		items = os.listdir(folder_path)
		for item in items:
			item_path = folder_path + '/' + item
			if not os.path.isdir(item_path):
				continue
			files = os.listdir(item_path)
			file_info.append(map(lambda x: folder + '/' + item + '/' + x, files))
	return file_info

def saveFileInfo(file_list, output_path):
	print "Writing file info to", output_path
	with open(output_path, 'w') as f:
		label = 0
		for items in file_list:
			for item in items:
				line = ''.join(item) + '\n'
				#line = ' '.join([item[item.find('lfw-cropped')+12:],str(label)]) + '\n'
				f.write(line)
			#label += 1
		f.close()
if __name__ == '__main__':
	root_path = 'F:/publicData/YouTubeFaces/frame_images_DB'
	file_info = loadInfo(root_path)
	output_path = 'F:/zhangli_code/face-recognition-video/data/YTF-frame_images.txt'
	saveFileInfo(file_info, output_path)

