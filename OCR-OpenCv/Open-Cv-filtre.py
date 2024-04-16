import cv2
import sys
import numpy as np


if sys.platform=='win32':
    deltax=0
    deltay=0
else:
    deltax=50
    deltay=105


konum="test.jpg"

def ana(goruntu_al):
    cam=cv2.VideoCapture(0)

    while(True):
        counter=0
        ret,frame=cam.read()
        if not ret:
            break
        cv2.imshow('goruntu',frame)
        cv2.moveWindow('goruntu',30,30)
        k=cv2.waitKey(30)
    
        counter+=1
        if counter==1:
            if goruntu_al==1:
                print('Görüntü Alınıyor....')
                

        if k == 27 or k == ord('q'):
            break
        
    cv2.imwrite('test.jpg',frame)#Kaydet

    if cam.isOpened():
        cam.release()
    cv2.destroyAllWindows()


def filtering():
    img=cv2.imread(konum)

    m=img.copy()
    m[:,:,1]=0
    m[:,:,2]=0

    y=img.copy()
    y[:,:,0]=0
    y[:,:,2]=0

    k=img.copy()
    k[:,:,0]=0
    k[:,:,1]=0

    print(img.shape)

    cv2.imshow('ORIJINAL',img)
    cv2.moveWindow('ORIJINAL',10,10)

    cv2.imshow('MAVI',m)
    cv2.moveWindow('MAVI',10,img.shape[0]+deltay)

    cv2.imshow('KIRMIZI',y)
    cv2.moveWindow('KIRMIZI',img.shape[1]+deltax,10)

    cv2.imshow('YESIL',k)
    cv2.moveWindow('YESIL',img.shape[1]+deltax,img.shape[0]+deltay)


    

    while True:
        m=cv2.waitKey(30)
        if m == 27 or m == ord('q'):
            break
    cv2.destroyAllWindows


def blur():
    img1=cv2.imread(konum)

    n=11
    kernel=np.ones((n,n),np.float32)/(n*n*1.0)

    blur=cv2.filter2D(img1,-1,kernel)

    cv2.imshow('img1',img1)
    cv2.imshow('filter',blur)
    cv2.moveWindow('img1',10,10)
    cv2.moveWindow('filter2D',img1.shape[1]+deltax,10)

    while True:
        n=cv2.waitKey(30)
        if n == 27 or n == ord('q'):
            break
    cv2.destroyAllWindows

def keskinleştirme():
    img2= cv2.imread(konum,0)
    kernel=np.ones((5,5),np.uint8)
    
    keskinleştirme_adımı=input("Keskinleştirme adım sayısını giriniz:")

    for i in range(1,20):
        erosion1=cv2.erode(img2,kernel,iterations=1) 
        dilation=cv2.dilate(erosion1,kernel,iterations=1)
    
    cv2.imshow('keskinlestirme',dilation)
    

    while True:
        t=cv2.waitKey(40)
        if t == 27 or t == ord('q'):
            break
    cv2.destroyAllWindows

def kamera_çözünürlüğü():
    cam=cv2.VideoCapture(0)
    çözünürlük=[(1320,1080),(1600,900),(1366,768),(1280,720)]

    for j in range(len(çözünürlük)):
        w0=int(çözünürlük[j][0])
        h0=int(çözünürlük[j][1])
        cam.set(3,w0)
        cam.set(4,h0)

def hat_tanıma():
    cam=cv2.VideoCapture(0)
    cam.set(3,1600)
    while True:
        
        alpha=2.5

        img3=cv2.imread(konum)
        gri=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
        blur=cv2.GaussianBlur(gri,(7,7),0)
        canny=cv2.Canny(blur,30,50)
        canny=cv2.bitwise_not(canny)

        canny=cv2.erode(canny,(7,7),iterations=1)
        canny=cv2.dilate(canny,(7,7),iterations=1)
        canny=cv2.convertScaleAbs(canny,alpha=alpha)
        kernel_local=np.array([[-1,-1,-1],
                               [-1,9,-1],
                               [-1,-1,-1]])
        
        canny=cv2.filter2D(canny,-1,kernel_local)

        imaj=cv2.bitwise_and(gri,gri,mask=canny)

        cv2.imshow('imaj',imaj)
        cv2.moveWindow('imaj',10,10)
        cv2.imshow('canny',canny)
        cv2.moveWindow('canny',imaj.shape[1]+deltax,10)

        m=cv2.waitKey(40)
        if m == 27 or m == ord('q'):
            break



if __name__=="__main__":
    ana(1)
    filtering()
    blur()
    keskinleştirme()
    hat_tanıma()
