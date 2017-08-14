import numpy as np
import cv2
from matplotlib import pyplot as plt


def generate_histogram(image):

	width, height = image.shape
	histogram = np.zeros(256)

	for i in range(width):
		for j in range(height):
			histogram[image[i,j]] += 1

	return histogram

	
def equalize_histogram(image):

	histogram = generate_histogram(image)
	width, height = image.shape
	pixels = width*height
	
	probabilities = []
	cumulative_probabilities = [0,]
	result = []

	for intensity_level in histogram:
		probabilities.append(intensity_level/pixels)
	
	for probability in probabilities:
		cumulative_probabilities.append(probability + cumulative_probabilities[-1])
	
	cumulative_probabilities.pop(0)
	for c_p in cumulative_probabilities:
		result.append(int(c_p*255))
	
	for i in range(width):
		for j in range(height):
			image[i,j] = result[image[i,j]]
	
	generate_histogram(image)
	return image


def median_filter(image):
    width, height = image.shape
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            pixel_intensities = [image[x, y], image[x, y - 1], image[x, y + 1], image[x - 1, y], image[x + 1, y],
                                 image[x - 1, y - 1],
                                 image[x - 1, y + 1], image[x + 1, y - 1], image[x + 1, y + 1]]
            pixel_intensities.sort()
            image[x, y] = pixel_intensities[4]
    return image


image2 = cv2.imread('466-cw3image.jpg', 0)
image2 = median_filter(image2)
image2 = cv2.fastNlMeansDenoising(image2, None, 10, 7, 21)

image4 = np.zeros((image2.shape[0], image2.shape[1]+2), np.uint8)
image5 = cv2.filter2D(image2, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))
for i in range(image2.shape[0]):
	for j in range(image2.shape[1]):
		image4[i,j] = image5[i,j]
image4[:,898:899] = image5[:,897:898]
image4[:,899:900] = image5[:,897:898]


cv2.imshow("Noise Removed", image2)
cv2.waitKey(0)

image2 = equalize_histogram(image2)

image3 = np.zeros((image2.shape[0], image2.shape[1]+2), np.uint8)
for i in range(image2.shape[0]):
	for j in range(image2.shape[1]):
		image3[i,j] = image2[i,j]
image3[:,898:899] = image3[:,897:898]
image3[:,899:900] = image3[:,897:898]

image = cv2.cvtColor(image4, cv2.COLOR_GRAY2BGR)

for i in range(1): # 4 ROW
	for j in range(2): # 3 COLUMN
		image3[i*175:(i+1)*175,j*300:(j+1)*300] = equalize_histogram(image3[i*175:(i+1)*175,j*300:(j+1)*300])

del image2
image2 = cv2.cvtColor(image3, cv2.COLOR_GRAY2BGR)
image2 = cv2.filter2D(image2, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))

mask = np.zeros(image2.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (0, 47, 716, 502)
cv2.grabCut(image2, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
image2 = image2 * mask2[:, :, np.newaxis]
image2 = cv2.erode(image2, np.ones((3,3), np.uint8))
image2 = cv2.dilate(image2, np.ones((3,3), np.uint8), iterations = 3)
for i in range(image2.shape[0]):
	for j in range(image2.shape[1]):
		if image2[i,j,0] != 0:
			image2[i,j] = (0,0,255)

cv2.addWeighted(image2, 0.3, image, 0.7, 0, image)

cv2.imshow("Final", image)
cv2.waitKey(0)
