import cv2
from tensorflow.keras.preprocessing.image import img_to_array


class ImageToArray:
    def __int__(self, data_format=None):
        self.data_format = data_format

    def preprocess(self, image):
        image = img_to_array(image,data_format=self.data_format)
        return image