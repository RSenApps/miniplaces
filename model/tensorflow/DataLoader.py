import os
import numpy as np
import scipy.misc
import h5py
np.random.seed(123)

# loading data from .h5
class DataLoaderH5(object):
    def __init__(self, **kwargs):
        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']

        # read data info from lists
        self.f = h5py.File(kwargs['data_h5'], "r")
        self.im_set = np.array(self.f['images'])
        self.lab_set = np.array(self.f['labels'])

        self.num = self.im_set.shape[0]
        assert self.im_set.shape[0]==self.lab_set.shape[0], '#images and #labels do not match!'
        assert self.im_set.shape[1]==self.load_size, 'Image size error!'
        assert self.im_set.shape[2]==self.load_size, 'Image size error!'
        print('# Images found:', self.num)

        

        self.shuffle()
        self._idx = 0
        
    def next_batch(self, batch_size):
        labels_batch = np.zeros(batch_size)
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3)) 
        
        for i in range(batch_size):
            image = self.im_set[self._idx]
            image = image.astype(np.float32)/255.
            if self.randomize:
                #bg_value = np.median(image)
                
                #shiftx = np.random.randint(-10, 10, 1)
                #shifty = np.random.randint(-10, 10, 1)
                #image = scipy.ndimage.shift(image,[shiftx, shifty, 0], cval=bg_value)
            
                s_vs_p = 0.5
                amount = 0.001
                #out = np.copy(image)
                # Salt mode
                num_salt = np.ceil(amount * image.size * s_vs_p)
                coords = [np.random.randint(0, j - 1, int(num_salt)) for j in image.shape]
                image[coords] = 1

                image = image - self.data_mean

                #angle = np.random.randint(-5,5,1)
                #image = scipy.misc.imrotate(image,angle)
            
                if (np.random.randint(0, 1, 1)):
                    image = np.flip(image)

                crop = np.random.randint(70, 120, 1)[0]
                startx = image.shape[1]/2-(crop/2)
                starty = image.shape[0]/2-(crop/2)
                image = image[starty:starty+crop,startx:startx+crop, :]
                images_batch[i, ...] = scipy.misc.imresize(image, (self.fine_size,self.fine_size))

                #offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                #offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                image = image - self.data_mean
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2
                images_batch[i, ...] = image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            labels_batch[i, ...] = self.lab_set[self._idx]
            
            self._idx += 1
            if self._idx == self.num:
                #c = len(self.f['images']) / self.batch_count
                #self.current_data_batch = (self.current_data_batch + 1) % self.batch_count
                #self.im_set = np.array(self.f['images'][c * self.current_data_batch: (c+1) * self.current_data_batch])
                #self.lab_set = np.array(self.f['labels'][c * self.current_data_batch: (c+1) * self.current_data_batch])

                self._idx = 0
                if self.randomize:
                    self.shuffle()
        
        return images_batch, labels_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0

    def shuffle(self):
        perm = np.random.permutation(self.num)
        self.im_set = self.im_set[perm] 
        self.lab_set = self.lab_set[perm]

# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.data_root = os.path.join(kwargs['data_root'])

        # read data info from lists
        self.list_im = []
        self.list_lab = []
        with open(kwargs['data_list'], 'r') as f:
            for line in f:
                path, lab =line.rstrip().split(' ')
                self.list_im.append(os.path.join(self.data_root, path))
                self.list_lab.append(int(lab))
        self.list_im = np.array(self.list_im, np.object)
        self.list_lab = np.array(self.list_lab, np.int64)
        self.num = self.list_im.shape[0]
        print('# Images found:', self.num)

        # permutation
        perm = np.random.permutation(self.num) 
        self.list_im[:, ...] = self.list_im[perm, ...]
        self.list_lab[:] = self.list_lab[perm, ...]

        self._idx = 0
        
    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3)) 
        labels_batch = np.zeros(batch_size)
        for i in range(batch_size):
            image = scipy.misc.imread(self.list_im[self._idx])
            image = scipy.misc.imresize(image, (self.load_size, self.load_size))
            image = image.astype(np.float32)/255.
            image = image - self.data_mean
            if self.randomize:
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    image = image[:,::-1,:]
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            images_batch[i, ...] =  image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            labels_batch[i, ...] = self.list_lab[self._idx]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
        
        return images_batch, labels_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0
