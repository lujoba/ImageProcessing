
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

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

        # return np.hstack([(f.Compute(data)) for f in self.apply])
        return [(f.Compute(data)) for f in self.apply]

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


class Flips(Apply):
    """
    Flip the image on the horizontal plane
    """
    
    def __init__(self, orientation = 'both'):
        self.dimPerImg = 1

        self.orientation = orientation
       
    def Compute(self, image):

        if(self.orientation == 'horizontal'):
            self.orientation = 0
        elif(self.orientation == 'vertical'):
            self.orientation = 1
        elif(self.orientation =='both'):
            self.orientation = -1

        image = cv2.imread(image)
        return cv2.flip(image,self.orientation)    


class Inverse(Apply):
    """
    Mirror the colors of the image
    """
    
    def __init__(self):
        self.dimPerImg = 1
    
    def Compute(self, image):
        image = cv2.imread(image)
        y = 255 - image

        print(image.shape, y.shape)

        return y


class Scaling(Apply):
    """
    Scaling the image of the image
    """


class Rotation(Apply):
    """
    Rotate the image
    """

    def __init___(self):
        pass

    def Compute(self, image):
        pass


class SobelDerivative(Apply):
    """
    Calculate the derivatives of the image.
    """

    def __init__(self):
        self.dimPerImg = 1

    def Compute(self, image):
        y = []
        
        image = cv2.imread(image)

        y = cv2.Sobel(image,
                           cv2.CV_16S,
                           1,
                           0,
                           ksize = 3,
                           scale = 1,
                           delta = 0,
                           borderType = cv2.BORDER_DEFAULT)

        return y


class ScharrDerivative(Apply):
    """
    Detect the second derivatives of an image in horizontal and vertical directions.
    """
    
    def __init__(self):
        self.dimPerImg = 1

    def Compute(self, image):
        image = cv2.imread(image)
        y = cv2.GaussianBlur(image, (3, 3), 0)

        return y

class Laplacian(Apply):
    
    def __init__(self, kernel_size=3,scale=1, delta=0, borderType=cv2.BORDER_DEFAULT, ddepth = cv2.CV_16S):
        self.dimPerImg = 1
        self.kernel_size = kernel_size
        self.scale = scale
        self.delta = delta
        self.borderType= borderType
        self.ddepth = ddepth

    def Compute(self, image):
        image = cv2.imread(image)

        y = cv2.Laplacian(image,ddepth=self.ddepth,ksize=self.kernel_size, scale = self.scale, delta=self.delta, borderType=self.borderType)

        return y

class Blur(Apply):

    def __init__(self, kernel_size = 0, step_size = 0):
        self.dimPerImg = 1
        self.kernel_size = kernel_size
        self.step_size = step_size

    def Compute(self, image):
        image = cv2.imread(image)
        y = cv2.blur(image, (self.kernel_size, self.kernel_size), 0)

        return y

class GaussianBlur(Apply):
    
    def __init__(self, kernel_size = 0, step_size = 0):
        self.dimPerImg = 1
        self.kernel_size = kernel_size
        self.step_size = step_size

    def Compute(self, image):
        image = cv2.imread(image)
        y = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)
        
        return y

class MedianBlur(Apply):

    def __init__(self, kernel_size = 0, step_size = 0):
        self.dimPerImg = 1
        self.kernel_size = kernel_size
        self.step_size = step_size

    def Compute(self, image):
        image = cv2.imread(image)
        y = cv2.medianBlur(image, self.kernel_size) 

        return y

class BilateralBlur(Apply):

    def __init__(self, kernel_size = 0, step_size = 0):
        self.dimPerImg = 1
        self.kernel_size = kernel_size
        self.step_size = step_size

    def Compute(self, image):
        image = cv2.imread(image)
        y = cv2.bilateralFilter(image, self.step_size, self.step_size*2, self.step_size/2)

        return y

if __name__ == "__main__":
    ImageFilters = ImageFilters([GaussianBlur(kernel_size=5, step_size=2),
                                 Inverse(),
                                 Flips(),
                                 SobelDerivative(),
                                 ScharrDerivative(),
                                 Laplacian(), 
                                 Blur(kernel_size=3, step_size=2), 
                                 GaussianBlur(kernel_size=3, step_size=2),
                                 MedianBlur(kernel_size=3, step_size=2), 
                                 BilateralBlur(kernel_size=3, step_size=2)
                                ],1)
                                 # GaussianBlur(kernel_size=3, step_size=2)], 109)

    #path = '/home/pinazawa/gitProjs/datasets/NFeImg/Dataset_Nfe/notas_imagens/'
    path = '/home/pinazawa/Downloads/image.jpg'

    #filelist = os.listdir(path)

    # filelist = [path + x for x in filelist]

    # imageList2 = []
    # for i in tqdm(filelist):
        # f = None
        # f = cv2.imread(i)
        # imageList2.append(ImageFilters.Process(i))
        # print(ImageFilters.Process(i))
    a = ImageFilters.Process(path)

    b = plt.subplot(2, 1, 2)

    for image in a:
        b.imshow(image)
        plt.pause(1)
            
    b.show()


    # a = a.astype(int)
    #print(a.shape)
    # a.reshape(a[0],a[1])
    
    # f = cv2.imread(filelist[1])
    # imageList2.append(ImageFilters.Process(f))

    # imageList2 = np.array(imageList2) 