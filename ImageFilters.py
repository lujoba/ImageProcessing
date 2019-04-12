
import numpy as np
import cv2


class ImageFilters(object):

    def __init__(self, apply, nImg):
        super(ImageFilters, self).__init__()
        self.apply = apply
        self.nImg = nImg
        self.nApply = nImg * sum(
            [f.dimPerImg for f in self.apply])
        self.output = np.zeros(self.nImg * self.nApply)

    def Process(self, data):
        # TODO use pre-allocated output array instead of hstack

        return np.hstack([f.Compute(data) for f in self.apply])

    def __repr__(self):
        return "%s.%s(%s)" % (
            self.__class__.__module__,
            self.__class__.__name__,
            str([str(f) for f in self.apply])
        )


class Apply(object):

    def __repr__(self):
        return "%s.%s()" % (
            self.__class__.__module__,
            self.__class__.__name__
        )


class HorizontalFlips(Apply):
    """
    Flip the image on the horizontal plane
    """
    
    def __init__(self):
        self.dimPerImg = 1
    


class Inverse(Apply):
    """
    Mirror the colors of the image
    """
    
    def __init__(self):
        self.dimPerImg = 1
    
    def Compute(self, image):
        image = cv2.imread(image)
        y = [(255 - image)]

        return y


class Scaling(Apply):
    """
    Scaling the image of the image
    """


class Rotation(Apply):
    """
    Rotate the image
    """


class SobelDerivative(Apply):
    """
    Calculate the derivatives of the image.
    """

    def __init__(self):
        self.dimPerImg = 1

    def Compute(self, image):
        y = []
        
        image = cv2.GaussianBlur(image, (3, 3), 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray,
                           cv2.CV_16S,
                           1,
                           0,
                           ksize = 3,
                           scale = 1,
                           delta = 0,
                           borderType = cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray,
                           cv2.CV_16S,
                           0,
                           1,
                           ksize = 3,
                           scale = 1,
                           delta = 0,
                           borderType = cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        y.append(dst)

        return y


class ScharrDerivative(Apply):
    """
    Detect the second derivatives of an image in horizontal and vertical directions.
    """
    
    def __init__(self):
        self.dimPerImg = 1

    def Compute(self, image):
        y = []
        image = cv2.GaussianBlur(image, (3, 3), 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Scharr(gray, cv2.CV_16S, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_16S, 0, 1)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        dst = cv2.add(abs_grad_x, abs_grad_y)
        y.append(dst)

        return y

class Laplacian(Apply):
    
    def __init__(self):
        self.dimPerImg = 1

    def Compute(self, image):
        y = []

        image = cv2.GaussianBlur(image, (3, 3), 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3, scale=1, delta=0)
        dst = cv2.convertScaleAbs(gray_lap)
        y.append(dst)

        return y

class Blur(Apply):

    def __init__(self, kernel_size = 0.0, step_size = 0.0):
        self.dimPerChannel = 1
        self.kernel_size = kernel_size
        self.step_size = step_size

    def Compute(self, x):
        y = [cv2.blur(image, (i, i))
             for i in xrange(1, blurConfig['kernel_size'], blurConfig['step_size'])]
        
        return y

class GaussianBlur(Action):
    
    def __init__(self, kernel_size = 0.0, step_size = 0.0):
        self.dimPerChannel = 1
        self.kernel_size = kernel_size
        self.step_size = step_size

    def Compute(self, x):
        y = [cv2.GaussianBlur(image, (i, i), 0)
            for i in xrange(1, gaussianBlurConfig['kernel_size'], gaussianBlurConfig['step_size'])]

        return y

class MedianBlur(Action):

    def __init__(self, kernel_size = 0.0, step_size = 0.0):
        self.dimPerChannel = 1
        self.kernel_size = kernel_size
        self.step_size = step_size

    def Compute(self, x):
        y = [cv2.medianBlur(image, i)
            for i in xrange(1, medianBlurConfig['kernel_size'], medianBlurConfig['step_size'])]

        return y

class BilateralBlur(Action):

    def __init__(self, kernel_size = 0.0, step_size = 0.0):
        self.dimPerChannel = 1
        self.kernel_size = kernel_size
        self.step_size = step_size

    def Compute(self, x):
        y = [cv2.bilateralFilter(image, i, i*2, i/2)
            for i in xrange(1, bilateralBlurConfig['kernel_size'], bilateralBlurConfig['step_size'])]

        return y
