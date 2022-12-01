import os
import argparse
import pickle
import pandas as pd
import numpy as np
from PIL import Image, ImageOps

#Models
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from keras.applications.vgg16 import VGG16

class Captcha(object):
    def __init__(self):
        # Load preprocessors
        preprocessing_d = pickle.load(open('weights_and_labels.pkl', 'rb'))
        self.masks = preprocessing_d['masks']
        self.le = LabelEncoder()
        self.le.classes_ = np.load('classes.npy', allow_pickle=True)
        self.VGG_model = VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(40, 32, 3))
        
        for layer in VGG_model.layers:
            layer.trainable = False
        
        # Load Model
        self.model = xgb.XGBClassifier()
        self.model.load_model('XGB_IMDA.json')

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        processed_image = self._preprocessing(self._read_image(im_path))
        
        output = []
        
        for mask in self.masks:
            alpha = np.stack((processed_image[mask].reshape(10,8),)*3, axis=-1)
            alpha = np.array(Image.fromarray(np.uint8(alpha)).resize((32,40))) / 255.0

            feature_extractor = self.VGG_model.predict(alpha.reshape(1, 40, 32, 3))
            features = feature_extractor.reshape(feature_extractor.shape[0], -1)

            y_pred = self.model.predict(features)
            output.extend(self.le.inverse_transform(y_pred))
        
        self._write_output(''.join(output), save_path)
    
    def _read_image(self, im_path):
        img = Image.open(im_path)
        return img
    
    def _write_output(self, output, save_path):
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f'Contents in output file: {output}')
    
    def _preprocessing(self, img):
        g_image = ImageOps.grayscale(img)
        numpy_image = np.array(g_image)
        return numpy_image
    
def arg_parser():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Captcha Help Documentation')
    # Required positional argument
    parser.add_argument('--img_path', '--img-path', type=str,
                        help='Image input path')

    # Optional positional argument
    parser.add_argument('--output_path', '--output-path', type=str,
                        help='Identification output path')
    
    return parser
    

if __name__ == '__main__':
    # Parse Arguments
    parser = arg_parser()
    args = parser.parse_args()
    
    # Load Model
    model = Captcha()
    
    # Perform Inference
    model(args.img_path, args.output_path)