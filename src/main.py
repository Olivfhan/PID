import numpy as np
from PIL import Image
import cv2

def main():
    filename = 'placeholder.png'
    #loads image given filename
    im = Image.open(filename)
    #transforms to numpy array
    np_frame = np.array(im)
    #resizes to 512x512 if needed
    np_frame = cv2.resize(np_frame,dsize=(512,512), interpolation=cv2.INTER_CUBIC)
    pass

if __name__=="__main__":
    main()
