import os
import glob
import time

filepath = "D:/Downloads/AirSim/"

subsets = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]

subfolders = [d for d in os.listdir(os.path.join(filepath, subsets[0])) if os.path.isdir(os.path.join(filepath, subsets[0], d))]

   
ref_folder  = 'left'
pfm_list = ['depth_gt','depth_sgm','disparity_gt','disparity_sgm']

for i in range(0,len(subsets)):
	good_list = [f.split(".")[:-1] for f in os.listdir(os.path.join(filepath, subsets[i], ref_folder))]

	for j in range (0,len(subfolders)):
		file_list = os.listdir(os.path.join(filepath, subsets[i], subfolders[j]))
		ext = '.' + file_list[0].split('.')[-1]
		list = [f.split('.')[:-1] for f in file_list]
		for item in list:
			if item not in good_list:				
				print(os.path.join(filepath, subsets[i], subfolders[j],item[0]+ext))
				os.remove(os.path.join(filepath, subsets[i], subfolders[j],item[0]+ext))