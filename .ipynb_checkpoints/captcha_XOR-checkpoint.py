import os
import argparse
import pickle
import pandas as pd
import numpy as np
from PIL import Image, ImageOps

class Captcha(object):
    def __init__(self):
        # Load required masks and weights
        weights_and_labels = pickle.load(open('weights_and_labels.pkl', 'rb'))
        self.weights = weights_and_labels['XOR']
        self.labels = weights_and_labels['labels']
        self.masks = weights_and_labels['masks']

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
            pd_mask = (processed_image[mask] < 100)
            flattened_image = pd.Series(processed_image[mask]).where(~pd_mask, 1).where(pd_mask, 0)
            flattened_image = np.stack((flattened_image.values,)*len(self.labels), axis=0)
            y_results = (1 - np.sum(np.logical_xor(flattened_image, self.weights), axis=1) / 80)
            output.append(self.labels[np.argmax(y_results)])
        
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