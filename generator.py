import os.path
import json
#import scipy.misc
import numpy as np
import random
from skimage.transform import resize
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.shuffle = shuffle
        self.mirroring = mirroring
        self.rotation = rotation
        self.batch_size = batch_size
        self.image_size = image_size
        self.images = []
        self.labels = None

        with open(label_path) as f:
            labels_dict = json.load(f)

        key_list = list(np.arange(0, len(labels_dict)).astype('str'))
        self.labels = np.array(list(map(labels_dict.get, key_list)))

        for i in range(0, len(labels_dict)):
            filename = file_path + '/' + str(i) + '.npy'
            image = np.load(filename)
            image = resize(image, self.image_size)
            self.images.append(image)

        self.images = np.array(self.images)
        self.shuffling(self.shuffle)
        self.epochs = 0
        self.start_of_batch = 0
        self.end_of_batch = self.start_of_batch + batch_size

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #return images, labels

        self.augmente()

        if self.end_of_batch > len(self.labels):

            if self.start_of_batch == len(self.labels):
                self.start_of_batch = 0
                self.end_of_batch = self.batch_size
                self.epochs += 1
                self.shuffling(self.shuffle)

                batch_labels = self.labels[self.start_of_batch: self.end_of_batch]
                batch_images = self.images[self.start_of_batch: self.end_of_batch]
                self.start_of_batch = self.end_of_batch
                self.end_of_batch = self.start_of_batch + self.batch_size

            else:

                batch_labels = self.labels[self.start_of_batch : ]
                batch_images = self.images[self.start_of_batch: ]
                self.end_of_batch = self.batch_size - len(batch_labels)
                batch_labels = np.concatenate((batch_labels, self.labels[0: self.end_of_batch]))
                batch_images = np.concatenate((batch_images, self.images[0: self.end_of_batch]), axis=0)

                self.start_of_batch = 0
                self.end_of_batch = self.batch_size
                self.epochs += 1
                self.shuffling(self.shuffle)


        else:
            batch_labels = self.labels[self.start_of_batch: self.end_of_batch]
            batch_images = self.images[self.start_of_batch: self.end_of_batch]
            self.start_of_batch = self.end_of_batch
            self.end_of_batch = self.start_of_batch + self.batch_size


        batch = (batch_images, batch_labels)

        return batch


    def augmente(self):
        if self.mirroring:
            mirroring_mask = list(np.random.randint(0, 2, len(self.labels)).astype('bool'))
            self.images[mirroring_mask] = np.flip(self.images[mirroring_mask], axis=2)

        if self.rotation:
            rotations = np.random.randint(0, 4, size=len(self.labels))
            self.images = np.array(list(map(lambda image, rotation: np.rot90(image, k=rotation), self.images, rotations)))


    def shuffling(self, shuffle):
        if shuffle:
            index = np.random.permutation(len(self.labels))
            self.labels = self.labels[index]
            self.images = self.images[index]

    def current_epoch(self):
        # return the current epoch number
        return self.epochs

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        batch = self.next()
        plt_size = int(np.ceil(np.sqrt(self.batch_size)))
        fig, axes = plt.subplots(plt_size, plt_size)
        axes = axes.flatten()
        for i in range(self.batch_size):
            axes[i].imshow(batch[0][i])
            axes[i].set_title(self.class_name(batch[1][i]))
            axes[i].axis('off')

        for i in range(self.batch_size, plt_size * plt_size):
            axes[i].axis('off')

        fig.tight_layout()
        fig.show()



gen = ImageGenerator('exercise_data', 'Labels.json', 50, [32, 32, 3], rotation=False, mirroring=False,shuffle=True)
gen.show()
gen.show()
gen.show()
gen.show()
gen.show()




