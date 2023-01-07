import torch
import numpy as np

def mnist():
    
    dir = 'data/raw/corruptmnist/'
    train0 = np.load(dir+'train_0.npz')
    train1 = np.load(dir+'train_1.npz')
    train2 = np.load(dir+'train_2.npz')
    train3 = np.load(dir+'train_3.npz')
    train4 = np.load(dir+'train_4.npz')
    test = np.load(dir+'test.npz')
    
    train_img_concat = np.concatenate( (train0['images'].astype('float32'), 
                                    train1['images'].astype('float32'),
                                    train2['images'].astype('float32'),
                                    train3['images'].astype('float32'),
                                    train4['images'].astype('float32')) )
    train_images = torch.from_numpy(train_img_concat)
    test_images = torch.from_numpy(test['images'].astype('float32'))

    train_lab_concat = np.concatenate( (train0['labels'], 
                                        train1['labels'],
                                        train2['labels'],
                                        train3['labels'],
                                        train4['labels']) )
    train_labels = torch.from_numpy(train_lab_concat).view([1,-1])
    test_labels = torch.from_numpy(test['labels']).view([1,-1])

    train = {'images':train_images, 'labels':train_labels}
    test = {'images':test_images, 'labels':test_labels}
    
    return train, test

mnist()