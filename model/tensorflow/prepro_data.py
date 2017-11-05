import os
import numpy as np
import h5py
import scipy.misc
import scipy.ndimage
from PIL import Image
np.random.seed(123)
def createH5(params):

	# create output h5 file
	output_h5 = '%s_%d_%s.h5' %(params['name'], params['img_resize'], params['split'])
	f_h5 = h5py.File(output_h5, "w")

	# read data info from lists
	list_im = []
	list_lab = []
	with open(params['data_list'], 'r') as f:
	    for line in f:
	        path, lab =line.rstrip().split(' ')
	        list_im.append(os.path.join(params['data_root'], path))
	        list_lab.append(int(lab))
	list_im = np.array(list_im, np.object)
	list_lab = np.array(list_lab, np.uint8)
	N = list_im.shape[0]
	#augN = N
	#if (params['split'] == 'train'):
    #  		augN *= 10; #data augmentation
	print('# Images found:', N)
	
	# permutation
	perm = np.random.permutation(N) 
	list_im = list_im[perm]
	list_lab = list_lab[perm]

	im_set = f_h5.create_dataset("images", (augN,params['img_resize'],params['img_resize'],3), dtype='uint8') # space for resized images
	f_h5.create_dataset("labels", dtype='uint8', data=list_lab)

	for i in range(augN):
		image = scipy.misc.imread(list_im[i % N])
		assert image.shape[2]==3, 'Channel size error!'
		'''
		bg_value = np.median(image)
		angle = np.random.randint(-15,15,1)
		image = scipy.misc.imrotate(image,angle)
		shiftx = np.random.randint(-10, 10, 1)
		shifty = np.random.randint(-10, 10, 1)
		image = scipy.ndimage.shift(image,[shiftx, shifty, 0], cval=bg_value)
    
		s_vs_p = 0.5
		amount = 0.001
		out = np.copy(image)
		# Salt mode
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = [np.random.randint(0, j - 1, int(num_salt)) for j in image.shape]
		image[coords] = 1
    
		if (np.random.randint(0, 1, 1)):
			image = np.flip(image)

		crop = np.random.randint(70, 120, 1)[0]
		startx = image.shape[1]/2-(crop/2)
    		starty = image.shape[0]/2-(crop/2)
    		image = image[starty:starty+crop,startx:startx+crop, :]
    		'''
    		image = scipy.misc.imresize(image, (params['img_resize'],params['img_resize']))
    		#img = Image.fromarray(image, 'RGB')
    		#img.show();

		im_set[i] = image

		if i % 1000 == 0:
			print('processing %d/%d (%.2f%% done)' % (i, augN, i*100.0/augN))

	f_h5.close()

if __name__=='__main__':
	params_train = {
		'name': 'miniplaces',
		'split': 'train',
		'img_resize': 128,
		'data_root': '../../../images/',	# MODIFY PATH ACCORDINGLY
    		'data_list': '../../data/train.txt'
	}

	params_val = {
		'name': 'miniplaces',
		'split': 'val',
		'img_resize': 128,
		'data_root': '../../../images/',	# MODIFY PATH ACCORDINGLY
    		'data_list': '../../data/val.txt'
	}

	createH5(params_train)
	createH5(params_val)
