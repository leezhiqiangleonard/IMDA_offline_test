# IMDA_offline_test
*This challenge is done in conjunction with IMDA's offline test for the interview role of Data Scientist*

## Introduction
(A) Task 
Note: No advanced computer vision background is required to solve this challenge. A simple understanding of the 256 x 256 x 256 RGB color space is sufficient.

A website uses Captchas on a form to keep the web-bots away. However, the captchas it generates, are quite similar each time:
- the number of characters remains the same each time  
- the font and spacing is the same each time  
- the background and foreground colors and texture, remain largely the same
- there is no skew in the structure of the characters.  
- the captcha generator, creates strictly 5-character captchas, and each of the characters is either an upper-case character (A-Z) or a numeral (0-9).

Here, take a look at some of the captcha images on the form. As you can see, they resemble each other very much - just that the characters on each of them are different.

![alt text](https://github.com/leezhiqiangleonard/IMDA_offline_test/blob/main/sampleCaptchas/input/input00.jpg)
![alt text](https://github.com/leezhiqiangleonard/IMDA_offline_test/blob/main/sampleCaptchas/input/input01.jpg)
![alt text](https://github.com/leezhiqiangleonard/IMDA_offline_test/blob/main/sampleCaptchas/input/input02.jpg)
![alt text](https://github.com/leezhiqiangleonard/IMDA_offline_test/blob/main/sampleCaptchas/input/input03.jpg)
			
You are provided a set of twenty-five captchas, such that, each of the characters A-Z and 0-9 occur at least once in one of the Captchas' text. From these captchas, you can identify texture, nature of the font, spacing of the font, morphological characteristics of the letters and numerals, etc. Download this sample set from here for the purpose of creating a simple AI model or algorithm to identify the unseen captchas.

## Analysis

Full Analysis done in Jupyter notebook: [IMDA_analysis.ipynb](https://github.com/leezhiqiangleonard/IMDA_offline_test/blob/main/IMDA_analysis.ipynb)

Thought Process (Summarized):
1. Each captcha is very similar to each other with only differing alphabets/numbers. We extract the alphabets and numbers using masking method since Captcha images are in a fixed format (e.g. distance between each alphabet/numbers are always the same; font sizes are always the same). Hence, masking method can accurately extract the individual alphabet/numbers from the Captcha image. 
2. For classification, we can apply image generation by sampling each pixel for the alphabets/numbers individually, before using feature extraction (from the VGG16 model) as input for the XGBoost classifier. The classifer will in turn predict alphabet/number that is the closest match in the full dictionary of all alphabets and numbers. 
3. For XOR method (which is a logical "exclusive OR" function in python), we can binarize the pixels to 0 or 1 by thresholding values above 100 and below 100 respectively. After which, we can perform XOR elementwise against the full dictionary of all the alphabets and numbers (where vectors are obtained through all input examples A-Z0-9). Lastly, with the output ranging from 0 to 1 (with 0 being the exact match, and 1 being a completely wrong match), we derive the similarity distances between the input data and all alphabets and numbers. 

Methods used:
1. VGG16 feature extraction + XGBoost classification + Image generation
2. XOR function similarity

Brief Steps:
1. Grayscale image
2. Extract alphabets/numbers using masks
3. Apply methods above

## Evaluation of the best method: XOR Method. 
1. Computationally more efficient than the classification method.
2. Results are easier to interpret and explain for, as the output provides all the distances between each input element against the full dictionary of alphabets and numbers. In other words, based on the results, we are able to identify problematic input elements that are causing wrong predictions. Conversely, the classification method is a blackbox, and it is difficult to identify what went wrong based on the output. 


Please run example as follows:

```bash
python3 captcha_XOR.py --img-path <insert image path> --output-path <insert output path (.txt file)>
```

## How to run (all methods)

```bash
python3 captcha_XOR.py --img-path <insert image path> --output-path <insert output path (.txt file)>

python3 captcha_XGB.py --img-path <insert image path> --output-path <insert output path (.txt file)>
```
