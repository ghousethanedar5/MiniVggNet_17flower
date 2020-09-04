import os
import cv2
import numpy as np


class LoadDataset:
    def __int__(self, preprocessor=None):
        self.preprocessor = preprocessor
        if self.preprocessor is None:
            self.preprocessor = []

    def load(self, imagepaths, verbose=-1):
        data, labels = [], []
        for i, imagepath in enumerate(imagepaths):
            image = cv2.imread(imagepath)
            label = imagepath.split(os.path.spe)[-2]
            if self.preprocessor is not None:
                for p in self.preprocessor:
                    image = p.preprocess(image)
            data.append(image)
            labels.append(label)
        return ( np.array(data), np.array(labels) )
