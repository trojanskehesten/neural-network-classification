import cv2
import numpy as np
from os.path import isfile, join
from os import listdir
import random
from shutil import copyfile
import os
import pickle
random.seed(0)

def seperateData(data_dir):
    for filename in listdir(data_dir):
        if isfile(join(data_dir, filename)):
            tokens = filename.split('.')
            if tokens[-1] == 'png':
                image_path = join(data_dir, filename)
                if not os.path.exists(join(data_dir, tokens[0])):
                    os.makedirs(join(data_dir, tokens[0]))
                copyfile(image_path, join(join(data_dir, tokens[0]), filename))
                os.remove(image_path)


class DataSetGenerator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_labels = self.get_data_labels()
        self.data_info = self.get_data_paths()

    def get_data_labels(self):
        data_labels = []
        for filename in listdir(self.data_dir):
            if not isfile(join(self.data_dir, filename)):
                data_labels.append(filename)
        return data_labels

    def get_data_paths(self):
        data_paths = []
        for label in self.data_labels:
            img_lists=[]
            path = join(self.data_dir, label)
            for filename in listdir(path):
                tokens = filename.split('.')
                if tokens[-1] == 'png':
                    image_path=join(path, filename)
                    img_lists.append(image_path)
            random.shuffle(img_lists)
            data_paths.append(img_lists)
        return data_paths

    # to save the labels its optional incase you want to restore the names from the ids 
    # and you forgot the names or the order it was generated 
    def save_labels(self, path):
        pickle.dump(self.data_labels, open(path,"wb"))

    def get_mini_batches(self, batch_size=10, allchannel=True): # image_size=(200, 200),
        images = []
        labels = []
        empty=False
        counter=0
        each_batch_size=int(batch_size/len(self.data_info))
        while True:
            for i in range(len(self.data_labels)):
                label = np.zeros(len(self.data_labels),dtype=int)
                label[i] = 1
                if len(self.data_info[i]) < counter+1:
                    empty=True
                    continue
                empty=False
                img = cv2.imread(self.data_info[i][counter])
                # img = self.resizeAndPad(img, image_size)
                if not allchannel:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
                images.append(img)
                labels.append(label)
            counter+=1

            if empty:
                break
            # if the iterator is multiple of batch size return the mini batch
            if (counter)%each_batch_size == 0:
                yield np.array(images,dtype=np.uint8), np.array(labels,dtype=np.uint8)
                del images
                del labels
                images=[]
                labels=[]

    def get_test_batch(self, allchannel=True): # image_size=(200, 200),
        images = []
        labels = []
        empty=False
        counter=0
        while True:
            for i in range(len(self.data_labels)):
                label = np.zeros(len(self.data_labels),dtype=int)
                label[i] = 1
                if len(self.data_info[i]) < counter+1:
                    empty=True
                    continue
                empty=False
                img = cv2.imread(self.data_info[i][counter])
                # img = self.resizeAndPad(img, image_size)
                if not allchannel:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
                images.append(img)
                labels.append(label)
            counter+=1

            if empty:
                break

        yield np.array(images,dtype=np.uint8), np.array(labels,dtype=np.uint8)
        del images
        del labels

        
if __name__=="__main__":
    seperateData("./train")