import numpy as np
import cv2
import scipy as sc
from scipy.linalg import lstsq
from scipy.linalg import svd
def compute_epipole(F):
    #Принимает фундаментальную матрицу стереосистемы, возвращает эпиполюс
    U, S, V = np.linalg.svd(F)
    e = V[-1, :]
    e = e / e[2]
    return e

def compute_matching_homographies(e2, Rot, K):
    #Принимает эпиполюс, матрицу поворота для стереосистемы, матрицу внутренних параметров, возварщает ректификационные гомографии
    h = K[0][2]
    w = K[1][2]
    T = np.array([[1,0,-K[0][2]],
              [0,1,-K[1][2]],
              [0,0,1]],dtype = np.float32)
    e2_p = T @ e2
    e2_p = e2_p / e2_p[2]
    e2x = e2_p[0]
    e2y = e2_p[1]
    if e2x >= 0:
        a = 1
    else:
        a = -1
    R1 = a * e2x / np.sqrt(e2x ** 2 + e2y ** 2)
    R2 = a * e2y / np.sqrt(e2x ** 2 + e2y ** 2)
    R = np.array([[R1, R2, 0], [-R2, R1, 0], [0, 0, 1]],dtype = np.float32)
    e2_p = R @ e2_p
    x = e2_p[0]
    G = np.array([[1, 0, 0], [0, 1, 0], [-1/x, 0, 1]])
    H2 = np.linalg.inv(T) @ G @ R @ T
    H1 = H2 @ K @ Rot @ np.linalg.inv(K)
    return H1, H2

def init_K(fx,fy,ox,oy):
    #Инициализация матрицы внутренних параметров
    return np.array([[fx,0.,ox],[0.,fy,oy],[0.,0.,1.]],dtype = np.float32)

def point_selector(event,x,y,flags,param):
    #Вспомогательная функция
    if event == cv2.EVENT_FLAG_LBUTTON:
        points.append([x,y])

def point_selection(img,mouse_callback = point_selector):
    #Принимает изображение, правой кнопкой мыши выбираются точки на изображении, возвращает массив отмеченных точек
    global points
    points = []
    cv2.namedWindow('image',cv2.WND_PROP_FULLSCREEN) 
    cv2.setMouseCallback('image',mouse_callback)
    while (1):
        cv2.imshow("image",img)
        if cv2.waitKeyEx()==32:
            break
    cv2.destroyAllWindows()
    return np.array(points,dtype=np.float32)

def stereo_calibration(object_points,image_points1,image_points2,K):
    #Стереокалибровка, возвращает матрицу поворота и сдвига стереосистемы
    retval,rvec1,tvec1 = cv2.solvePnP(object_points,image_points1,K,None,flags = cv2.SOLVEPNP_ITERATIVE)
    retval,rvec2,tvec2 = cv2.solvePnP(object_points,image_points2,K,None,flags = cv2.SOLVEPNP_ITERATIVE)
    r1 = np.array(cv2.Rodrigues(rvec1)[0])
    r2 = np.array(cv2.Rodrigues(rvec2)[0])
    R = np.dot(r1,r2.T)
    rvecs = np.zeros(shape = (2,3))
    rvecs[0] = [0,0,0]
    rvecs[1] = np.array((np.dot(-R,tvec2) + np.array(tvec1).reshape((3,1)))).reshape(3,)
    return rvecs,R

def compute_distance(imgp1,imgp2,num_images,K,rvecs,R):
    #Возвращает расстояние от положения первой камеры до обозначенной точки по координате с двух изображений
    univec = np.zeros(shape = (num_images,3)) 
    univec[0] = [(imgp1[0][0]-K[0][2])/K[0][0],(imgp1[0][1]-K[1][2])/K[1][1],1]
    univec[1] = [(imgp2[0][0]-K[0][2])/K[0][0],(imgp2[0][1]-K[1][2])/K[1][1],1]
    univec[1] = np.dot(R,univec[1])
    univec[1] = univec[1]/univec[1][2]
    gram = np.matmul(univec,univec.T)*(1/num_images) 
    b = np.array([np.dot(univec[i]/num_images,np.sum(rvecs,axis = 0)-num_images*rvecs[i]) for i in range(num_images)],
                 np.float32)
    gram[0][1]=-1*gram[0][1]
    gram[1][0]=-1*gram[1][0]
    solves = np.linalg.solve(gram,b)
    rit = [solves[i]*univec[i] for i in range(num_images)]  
    robj = (np.sum(rit,axis = 0)+np.sum(rvecs,axis = 0))/num_images
    return robj

def matrixT(tvec):
    #Возвращает матрицу сдвига
    T = np.array([[0,-tvec[2],tvec[1]],
              [tvec[2],0,-tvec[0]],
              [-tvec[1],tvec[0],0]],dtype = np.float32)
    return T

def computeEssentialMatrix(R,T):
    #Считает естественную матрицу по матрице поворота и матрице сдвига стереосистемы
    E = np.matmul(T,R)
    return E

def computeFundamentalMatrix(K,E):
    #Считает фундаментальную матрицу по матрице внутренних параметров и естественной матрице
    matk_inv = np.linalg.inv(K)
    F = np.matmul(np.matmul(matk_inv.T,E),matk_inv)
    return F

def nothing(x):
    pass

def SGBM_trackbar(img1,img2):
    #Создает интерактивный интерфейс, позволяет изменять параметры мэтчинга и отслеживать карту глубины
    cv2.namedWindow('disp',cv2.WND_PROP_FULLSCREEN)
    cv2.resizeWindow('disp',200,200)
    cv2.createTrackbar('numDisparities','disp',1,17,nothing)
    cv2.createTrackbar('blockSize','disp',5,50,nothing)
    cv2.createTrackbar('preFilterType','disp',1,1,nothing)
    cv2.createTrackbar('preFilterSize','disp',2,25,nothing)
    cv2.createTrackbar('preFilterCap','disp',5 ,62,nothing)
    cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
    cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
    cv2.createTrackbar('speckleRange','disp',0,100,nothing)
    cv2.createTrackbar('speckleWindowSize','disp',3,25,nothing)
    cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
    cv2.createTrackbar('minDisparity','disp',5,25,nothing)
 
    stereo = cv2.StereoSGBM_create()

    while True:
        numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
        blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
        speckleRange = cv2.getTrackbarPos('speckleRange','disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
        minDisparity = cv2.getTrackbarPos('minDisparity','disp')

        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)
 
 
        disparity = stereo.compute(img1,img2)
 
        disparity = disparity.astype(np.float32)
 
        disparity = (disparity/16.0 - minDisparity)/numDisparities
 
        cv2.imshow("disp",disparity)
        if cv2.waitKey(1) == 32:
            cv2.destroyAllWindows()
            print("numDisparities:",numDisparities)
            print("blockSize:",blockSize)
            print("preFilterCap:",preFilterCap)
            print("uniquenessRatio:",uniquenessRatio)
            print("speckleRange:",speckleRange)
            print("speckleWindowSize:",speckleWindowSize)
            print("disp12MaxDiff:",blockSize)
            print("blockSize:",blockSize)
            print("minDisparity",minDisparity)
            return disparity
        
def image_index_tresholder(img,indices):
    #Возвращает облако точек по массиву 
    img1 = cv2.filter2D(img,0,np.array([0]))
    for i in indices:
        img1[int(i[1])-2:int(i[1])+1,int(i[0])-2:int(i[0])+1] = 255*np.ones(np.shape(img1[int(i[1])-2:int(i[1])+1,int(i[0])-2:int(i[0])+1]))
    return img1

def image_index_depth(img,indices):
    #Дополняет массив точек координатой глубины по карте глубины
    mass = []
    for i in indices:
        mass.append(img[int(i[1])][int(i[0])])
    mass = np.array(mass).reshape((-1,1))
    mass = np.append(indices,mass,axis = 1)
    return mass

    

def show_image(img):
    cv2.namedWindow("image",flags = cv2.WND_PROP_FULLSCREEN)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return