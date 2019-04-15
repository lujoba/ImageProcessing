from imgFil import ImageFilters as fil
import matplotlib.pyplot as plt
import cv2

# path for the image to be read
path = './image.jpeg'

# Creates two graphs for showing the original image and
# the transformed version
b = plt.subplot(2,1, 2)
c = plt.subplot(2,1,1)

# Read image in the path
original = cv2.imread(path)

#Choose which one of the transformations must be performed
ImageFilters = fil.ImageFilters([# fil.GaussianBlur(kernel_size=5, step_size=2),
                                 # fil.Inverse(),
                                 # fil.Flips(),
                                 # fil.SobelDerivative(),
                                 # fil.ScharrDerivative(),
                                 # fil.Laplacian(), 
                                 # fil.Blur(kernel_size=3, step_size=2), 
                                 # fil.GaussianBlur(kernel_size=3, step_size=2),
                                 # fil.MedianBlur(kernel_size=3, step_size=2), 
                                 # fil.BilateralBlur(kernel_size=3, step_size=2),
                                 # fil.Scaling(scaleXY = 1.5),
                                 fil.Rotation(angle = 45)
                                 ],1)

# Process the image. The output will be several new images
a = ImageFilters.Process(path)

# Here you can do whatever helps you with the transformed data...
# for example, plotting them. Remember to invert the RGB format to 
# GBR
for image in a:

	c.imshow(original[...,::-1])
	b.imshow(image[...,::-1])
	plt.show()

