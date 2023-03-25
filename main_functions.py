import numpy as np
import cv2
import scipy as sc
from scipy.linalg import lstsq
from scipy.linalg import svd
def compute_epipole(F):
    U, S, V = np.linalg.svd(F)
    e = V[-1, :]
    e = e / e[2]
    return e

def compute_matching_homographies(e2, Rot, K):
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
    return np.array([[fx,0.,ox],[0.,fy,oy],[0.,0.,1.]],dtype = np.float32)

def point_selector(event,x,y,flags,param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        points.append([x,y])

def point_selection(img,mouse_callback = point_selector):
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

def matT(tvec):
    T = np.array([[0,-tvec[2],tvec[1]],
              [tvec[2],0,-tvec[0]],
              [-tvec[1],tvec[0],0]],dtype = np.float32)
    return T

def computeEssentMatx(R,T):
    E = np.matmul(T,R)
    return E

def computeFundMat(K,E):
    matk_inv = np.linalg.inv(K)
    F = np.matmul(np.matmul(matk_inv.T,E),matk_inv)
    return F
