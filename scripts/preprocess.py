import torch
import torchvision.transforms as transforms
import numpy as np  
import cv2


class PreProcessImage(object):

    
    def __call__(self, *args, **kwds):
        image = np.array(image) 


        # B. COLOR SPACE CONVERSION (RGB -> YUV)
        # YUV separates brightness (Y) from color (UV), helping with shadows
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        
        # C. GAUSSIAN BLUR
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # D. RESIZE
        # Resize to NVIDIA standard (200 width, 66 height)
        
        # E. NORMALIZE
        # Normalize to [0, 1] range
        image = image / 255.0 
        image = image.astype(np.float32)
        return image
