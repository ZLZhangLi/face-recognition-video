#! /usr/bin/python
#-*- coding:utf-8 -*-
import os, random
# Iterate file system.
def loadInfo(dataset_root):
	folders = sorted(os.listdir(dataset_root))
	file_info = []
	for folder in folders:
		items = os.listdir(dataset_root + '/' + folder)
		for item in items:
			folder_path = dataset_root + '/' + folder + '/' + item
			if not os.path.isdir(folder_path):
				continue
			files = sorted(os.listdir(folder_path))
			file_info.append(map(lambda x: folder + '/' + item + '/' + x, files))
	return file_info

# Generate sample pairs.
def getPair(file_info, num_same = 0):
	num_class = len(file_info)
	sample_same = []	
	for i in xrange(num_class):
		while True:
			if len(file_info[i]) > 1:
				break
		sample1, sample2 = random.sample(file_info[i], 2)
		sample_same.append(sample1)
		sample_same.append(sample2)
	return (sample_same)

# Save pair info to file.
def savePair(pairs, output_path):
	with open(output_path, 'w') as f:
		for pair in pairs:
			line = ''.join(pair) + '\n'
			f.write(line)
		f.close()

if __name__ == '__main__':
	dataset_root = 'F:/publicData/YouTubeFaces/frame_images_DB'
	print('Iterating file system...')
	file_info = loadInfo(dataset_root)
	print('Done!')
	# Count sample and class number.
	num_class = len(file_info)
	num_sample = 0
	for i in xrange(num_class):
		num_sample += len(file_info[i])
	print('%d classes with %d images found.' %(num_class, num_sample))

	# Set pair number.
	num_same = num_sample

	# Generate sample pairs.
	print('Generating sample pairs...')
	sample_same = getPair(file_info, num_same)
	print('%d same-class pairs generated.' %(num_same))

	# Set output file.
	dataset_name = 'frame_images'
	file_sample_same = 'data/sample_same_' + dataset_name + '.txt'

	# Save pair info to file.
	print('Saving sample pairs to file...')
	savePair(sample_same, file_sample_same)
	print('Done!')

