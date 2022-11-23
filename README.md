# IMDA_offline_test
*This challenge in done in conjunction with IMDA's offline test for the role of Data Scientist*

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

Analysis done in Jupyter notebook: [IMDA_analysis.ipynb](https://github.com/leezhiqiangleonard/IMDA_offline_test/blob/main/IMDA_analysis.ipynb)

Summary: Each captcha is very similar to each other with only differing alphabets/numbers. We extract the alphabets and numbers using masking method since each position of the alphabets and numbers are positioned the same. For classification, we apply image generation by sampling each pixel for each alphabet and number individually before using feature extraction as input for the classifier. The output predicts the most likely alphabet. For XOR method, we binarize the pixels to 0 or 1 by thresholding values above 100 and below 100 respectively. Then we perform XOR elementwise against all the alphabets and numbers (Vectors are obtained through all input examples A-Z0-9) and calculating the error and taking 1 minus that error. Exact matches outputs a value of 1. We then obtain the respective label corresponding to the index with that value of 1.

Methods:
1. VGG16 feature extraction + XGBoost classification + Image generation
2. XOR function similarity

Brief Steps:
1. Grayscale image
2. Extract alphabets/numbers using masks
3. Apply methods above

## Method of choice (For this case): XOR Method
Reason: Simple, efficient, computes faster, and more reliable as structure and font would not change.

Please run example as follows:

```bash
python3 captcha_XOR.py --img-path <insert image path> --output-path <insert output path (.txt file)>
```

## How to run (all methods)

```bash
python3 captcha_XOR.py --img-path <insert image path> --output-path <insert output path (.txt file)>

python3 captcha_XGB.py --img-path <insert image path> --output-path <insert output path (.txt file)>
```
