from utils_img import *
import cv2
import numpy as np

def process( image, size = 256, crop = 'normal', preprocessing = 'clahe', vessel_model=None, fourth = None,  ):

	IMAGE_SIZE = 512

	if crop == 'normal':
		image = normal_crop(image, IMAGE_SIZE)

	elif crop == 'circle':
		image = circle_crop(image, IMAGE_SIZE)

	if preprocessing == 'clahe':
		preprocessed_image = clahe(image)
		preprocessed_image = cv2.resize(preprocessed_image, (size,size), interpolation = cv2.INTER_NEAREST)

	elif preprocessing == 'ben':
		preprocessed_image = ben(image)
		preprocessed_image = cv2.resize(preprocessed_image, (size,size), interpolation = cv2.INTER_NEAREST)

	elif preprocessing is None:
		preprocessed_image = image
		preprocessed_image = cv2.resize(preprocessed_image, (size,size), interpolation = cv2.INTER_NEAREST)

	if vessel_model is not None:
		heatmap = vessel_model.predict(image)
		heatmap = cv2.resize(heatmap, (size,size), interpolation = cv2.INTER_NEAREST)

		preprocessed_image = np.asarray(preprocessed_image)
		heatmap = np.asarray(heatmap)

		merge = np.dstack((preprocessed_image, heatmap))

	if vessel_model is None and fourth == 'opencv_veins':
		heatmap = extract_bv(image)
		heatmap = cv2.resize(heatmap, (size,size), interpolation = cv2.INTER_NEAREST)

		preprocessed_image = np.asarray(preprocessed_image)
		heatmap = np.asarray(heatmap)


		merge = np.dstack((preprocessed_image, heatmap))

	if vessel_model is None and fourth == 'krish':
		heatmap = Krish(image)
		heatmap = cv2.resize(heatmap, (size,size), interpolation = cv2.INTER_NEAREST)

		preprocessed_image = np.asarray(preprocessed_image)
		heatmap = np.asarray(heatmap)


		merge = np.dstack((preprocessed_image, heatmap))

	elif vessel_model is None and fourth is None:
		merge = np.asarray(preprocessed_image)

	return merge









	







