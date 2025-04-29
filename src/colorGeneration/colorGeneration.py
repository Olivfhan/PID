import cv2
import numpy as np
import matplotlib.pyplot as plt


def estructurant(radius):
    kernel = np.zeros((2*radius+1, 2*radius+1) ,np.uint8)
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1
    kernel[0,radius-1:kernel.shape[1]-radius+1] = 1
    kernel[kernel.shape[0]-1,radius-1:kernel.shape[1]-radius+1]= 1
    kernel[radius-1:kernel.shape[0]-radius+1,0] = 1
    kernel[radius-1:kernel.shape[0]-radius+1,kernel.shape[1]-1] = 1
    return kernel

def bwareaopen(image:np.array, size:int):
    output=image.copy()
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    for i in range(1,nlabels):
        regions_size=stats[i,4]
        if regions_size<size:
            output[labels == i] = 0
    return output

def morphologicalOpeningClosing(im):

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    kernel = estructurant(7)
    reconstructed = cv2.morphologyEx(im,cv2.MORPH_OPEN,kernel)
    reconstructed = cv2.morphologyEx(reconstructed,cv2.MORPH_CLOSE,kernel)
    return reconstructed

def SobelGradient(im):
    sobelx = cv2.Sobel(im,cv2.CV_16S,1,0,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
    sobely = cv2.Sobel(im,cv2.CV_16S,0,1,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    res = bwareaopen(combined,100)
    #edges = cv2.Canny(image=combined,threshold1=25,threshold2=200)
    #magnitude = np.sqrt(sobelx**2.0+sobely**2.0)
    #combined = cv2.dilate(combined,estructurant(3),1)
    return res

def binarize(im):
    _,res = cv2.threshold(im,16,255,cv2.THRESH_BINARY_INV)
    return res

def getRegions(im):
    _,markers = cv2.connectedComponents(im)
    return markers

def averageRegions(im,markers):
    res = []
    im = cv2.cvtColor(im,cv2.COLOR_BGR2LAB)
    for i in range(np.max(markers)):
        res.append([0,0,0,0])
    for i in range(len(markers)):
        for j in range(len(markers[0])):
            current = res[markers[i][j]-1]
            pixel = im[i][j]
            current[0]+=int(pixel[0])
            current[1]+=int(pixel[1])
            current[2]+=int(pixel[2])
            current[3]+=1
            res[markers[i][j]-1] = current
    for i in range(len(res)):
        current = res[i]
        div = current[3]
        current[0] = int(current[0])//div
        current[1] = int(current[1])//div
        current[2] = int(current[2])//div
        res[i] = current[:3]
    for i in range(len(im)):
        for j in range(len(im[0])):
            im[i][j]=res[markers[i][j]-1]
    im = cv2.cvtColor(im,cv2.COLOR_LAB2BGR)
    return im

def colorGeneration():
    im = cv2.imread('../../Images/raw/Angela_Lansbury_0002.jpg',cv2.IMREAD_GRAYSCALE)
    im_color = cv2.imread('../../Images/raw/Angela_Lansbury_0002.jpg',cv2.IMREAD_UNCHANGED)
    im2 = morphologicalOpeningClosing(im)
    im3 = SobelGradient(im2)
    im4 = binarize(im3)
    im5 = getRegions(im4)
    im6 = averageRegions(im_color.copy(),im5)
    plt.figure(figsize=(15,5))
    
    plt.subplot(2, 3, 1)
    #plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.imshow(cv2.cvtColor(im_color,cv2.COLOR_BGR2RGB))
    plt.title('Imagen de Entrada')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(im2,cmap='gray')
    plt.title('Tras cerrado y apertura')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(im3,cmap='gray')
    plt.title('Tras gradiente de Sobel')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(im4,cmap='binary')
    plt.title('Tras binarizado')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(im5,cmap='gray')
    plt.title('Regiones')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(im6,cv2.COLOR_BGR2RGB))
    plt.title('Coloreado')
    plt.axis('off')




    plt.show()
    
if __name__=="__main__":
    colorGeneration()
