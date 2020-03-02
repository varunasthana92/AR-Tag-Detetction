import matplotlib.pyplot as plt 
import numpy as np 
import sys
sys.path.remove(sys.path[1])
import cv2
import math
import argparse

def PMat(H):
	K= np.array([[1406.08415449821,0,0],
		[2.20679787308599, 1417.99930662800,0],
		[1014.13643417416, 566.347754321696,1]])
	K=K.transpose()
	inv_K = np.linalg.inv(K)
	B = np.matmul(inv_K, H)
	b1 = B[:, 0].reshape(3, 1)
	b2 = B[:, 1].reshape(3, 1)
	# r3 = np.cross(B[:, 0], B[:, 1])
	b3 = B[:, 2].reshape(3, 1)

	h1 = H[:, 0].reshape(3, 1)
	h2 = H[:, 1].reshape(3, 1)
	# h3 = H[:, 2].reshape(3, 1)

	lamb_da= 2/(np.linalg.norm(inv_K.dot(h1))+np.linalg.norm(inv_K.dot(h2)))
	t = lamb_da*b3
	r1 = lamb_da*b1
	r2 = lamb_da*b2
	# r3 = (r3 * lamb_da * lamb_da).reshape(3, 1)
	r3 = (np.cross(r1.reshape(-1), r2.reshape(-1))).reshape(3, 1)
	RT = np.concatenate((r1, r2, r3,t), axis=1)
	P= np.matmul(K,RT)
	return P


def tagMat(img):
	b_img= (img>200)*1
	h,w = b_img.shape[0], b_img.shape[1]
	# h_, w_ = int(h/8), int(w/8)
	sq_max_size= np.max(np.sum(b_img[20:h-20,20:w-20],axis=1))
	sq_grid_size= int(sq_max_size/4)
	h_, w_= sq_grid_size, sq_grid_size
	req_img_size= 8*h_
	if(h<req_img_size):
		pad= int((req_img_size-h)/2)
		new= np.zeros([req_img_size,req_img_size])
		new[pad:pad+h, pad:pad+w]= b_img
		b_img=new
	gridImg= np.zeros([8,8])
	# gridImg[2:6,2:6] =1
	# b_img[2*h_:3*h_, ]
	for i in range(2,6):
		for j in range(2,6):
			white= b_img[i*h_:(i+1)*h_, j*w_:(j+1)*w_].sum()
			black= (h_*w_)- white
			if(white>black):
				gridImg[i,j]= 1
			else:
				gridImg[i,j]= 0

	img= b_img*255
	pxstep= int(img.shape[1]/8)
	x, y = pxstep, pxstep
	while x < img.shape[1]:
		cv2.line(img, (x, 0), (x, img.shape[0]), [255,0,0],2)
		x += pxstep
	while y < img.shape[0]:
		cv2.line(img, (0, y), (img.shape[1], y), [255,0,0],2)
		y += pxstep
	cv2.imwrite('warp0.png', img)
	# print(gridImg[2:6,2:6])
	return gridImg[2:6,2:6]


def tagID(img):
	grid= tagMat(img)
	tl ,tr, br, bl = 0, 0, 0, 0
	tl, tr, br, bl = grid[0,0], grid[0,3], grid[3,3], grid[3,0]
	id= np.array([grid[2,1], grid[2,2], grid[1,2], grid[1,1]])
	angle = 0
	if(br):
		pass
	elif(tr):
		angle= 90
		id= np.roll(id,-1)
	elif(tl):
		angle = 180
		id= np.roll(id,-2)
	elif(bl):
		angle = 270
		id= np.roll(id,-3)

	return angle, id

def Homography(allTags, finalCords):
	Amat = []
	H_allTags = []
	for i in allTags:
		A= np.zeros([8,9])
		idx = 0
		Amat=[]
		for k in range(0,8,2):	# TL, TR, BR, BL
			x1 = i[int(k/2)][0]
			y1 = i[int(k/2)][1]
			x2 = finalCords[int(k/2)][1]
			y2 = finalCords[int(k/2)][0]
			A[k] = np.array([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
			A[k + 1] = np.array([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])
			Amat.append(A[k])
			Amat.append(A[k+1])
		_, _, V = np.linalg.svd(Amat, full_matrices=True)
		h = (V[8,:]/V[8,8]).reshape(3,3)
		H_allTags.append(np.array(h))
	return H_allTags

def checkSquare(img, poly):
	order=[]
	seq=[]
	fact= 5
	for cord in poly:
		count=0
		tl, tr, bl, br = 0, 0, 0, 0
		h= cord[0][1]
		w= cord[0][0]
		if(img[h - fact, w - fact]>=150):
			count+=1
			tl=1
		if(img[h + fact, w - fact]>=150):
			count+=1
			bl=1
		if(img[h - fact, w + fact]>=150):
			count+=1
			tr=1
		if(img[h + fact, w + fact]>=150):
			count+=1
			br=1
		if(count==3):
			if (tl and tr and bl):
				seq.append('TL')
			elif (tl and tr and br):
				seq.append('TR')
			elif(tr and br and bl):
				seq.append('BR')
			elif(tl and br and bl):
				seq.append('BL')
		else:
			return False, None
	order.append(seq)
	return True, order



def contoursFromThresholod(gray):
	ret,thresh = cv2.threshold(gray,127,255,0)
	return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def contoursFromCanny(gray):
	blur = cv2.blur(gray,(5	,5))
	thresh = cv2.Canny(blur, 150, 150)
	return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


Parser = argparse.ArgumentParser()
Parser.add_argument('--Video', default="../data/Tag0.mp4", help='Give path of the video')
Parser.add_argument('--Image', default="../data/Lena.png", help='Give path of the image you want to place on tag')
Args = Parser.parse_args()

VideoPath = Args.Video
ImagePath = Args.Image
# names = ["multipleTags.mp4","Tag0.mp4","Tag1.mp4","Tag2.mp4"]
cap = cv2.VideoCapture(VideoPath)
# names = ["multipleTags.mp4","Tag0.mp4","Tag1.mp4","Tag2.mp4"]

while (cap.isOpened()):	
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	contours, hierarchy = contoursFromThresholod(gray)
	
	allChildsID=[]
	allChilds= []
	allHier= []
	i= len(contours)-1
	while (i>=0):
		childCnt= contours[i]
		#hData= [Next, Previous, First_Child, Parent]
		parentId= hierarchy[0][i][3]
		if(parentId!= -1):
			allChildsID.append(i)
			parentCnt= contours[parentId]
			c_area= cv2.contourArea(childCnt)
			p_area= cv2.contourArea(parentCnt)
			if(c_area< (0.9*p_area)):
				allChildsID.append(i)
				grand_Pa_Id= hierarchy[0][parentId][3]
				if(grand_Pa_Id==-1):
					if(p_area<(2*c_area)):
						allChildsID.append(parentId)		
				else:
					grand_Pa_Cnt= contours[grand_Pa_Id]
					g_area= cv2.contourArea(grand_Pa_Cnt)
					if((g_area/p_area) <1.1):
						allChildsID.append(grand_Pa_Id)
						hierarchy[0][parentId][3]= -1
						hierarchy[0][grand_Pa_Id][3] = -1
		i-=1

	unique_idx = []
	for x in allChildsID:
		if x not in unique_idx:
			unique_idx.append(x)

	x= 0
	while(x<len(unique_idx)):
		idx= unique_idx[x]
		if(cv2.contourArea(contours[idx])<30000 and cv2.contourArea(contours[idx])>500):
			allChilds.append(contours[idx])
			allHier.append(hierarchy[0][idx])
		else:
			del unique_idx[x]
			x-=1
		x+=1

	all_tags=[]
	seq_points=[]
	for c in allChilds:
		peri = cv2.arcLength(c, True)
		quad = cv2.approxPolyDP(c, 0.1 * peri, True)
		if (len(quad) == 4):
			rslt, order = checkSquare(gray, quad)
			temp = np.unique(np.array(order))
			if(rslt and len(temp)==len(order[0])):
				seq_points.append(order)
				all_tags.append(quad)


	#####################################################################################################
	#####################################################################################################
	##### Sorting in defined order of TL, TR, BR, BL for HOMOGRAPHY, grid formation and tilt angle ######
	#####################################################################################################
	#####################################################################################################
	seq_tags=[]
	for i in range(len(all_tags)):
		first, second, third, forth = None, None, None, None
		for k in range(4):
			if(seq_points[i][0][k]== 'TL'):
				first= all_tags[i][k][0]
			elif(seq_points[i][0][k]== 'TR'):
				second= all_tags[i][k][0]
			elif(seq_points[i][0][k]== 'BR'):
				third= all_tags[i][k][0]
			elif(seq_points[i][0][k]== 'BL'):
				forth= all_tags[i][k][0]
		seq_tags.append(np.array([first, second, third, forth]))

	#### Homography ####
	tag_ref= cv2.imread("../data/ref_marker.png",0)
	imgH= tag_ref.shape[0]
	imgW= tag_ref.shape[1]
	tag_plane=[(0,0), (imgH-1,0), (imgW-1, imgH-1), (0,imgW-1)] # (h,w) in order o TL, TR, BR, BL
	H_allTags= Homography(seq_tags, tag_plane)

	## Warping the world frame to camera image plane ####
	i=-1
	frame1= frame.copy()
	for H in H_allTags:
		i+=1
		im_out= np.zeros([tag_ref.shape[0], tag_ref.shape[1]])
		Hinv = np.linalg.inv(H)
		
		P= PMat(Hinv)
		for m in range(tag_ref.shape[0]):
			for n in range(tag_ref.shape[1]):
			    h1, w1, z1 = np.matmul(P,[m,n,0,1])
			    ha, wa = int(h1/z1), int(w1/z1)
			    # cv2.circle(new, (ha,wa), 3, (0,0,255)) 
			    if (ha < gray.shape[0] and ha > 0 and wa < gray.shape[1] and wa > 0):
			        im_out[m][n]= gray[wa][ha]

		## Generating tag id
		angle, _ = tagID(im_out)
		
		t = int(angle/90)
		r_img= cv2.imread(ImagePath)
		r_img= cv2.resize(r_img, (tag_ref.shape[1],tag_ref.shape[0]))
		r_img= np.rot90(r_img,t)

		P= PMat(Hinv)
		for m in range(tag_ref.shape[0]):
			for n in range(tag_ref.shape[1]):
			    h1, w1, z1 = np.matmul(P,[m,n,0,1])
			    ha, wa = int(h1/z1), int(w1/z1)
			    # cv2.circle(new, (ha,wa), 3, (0,0,255)) 
			    if (ha < gray.shape[0] and ha > 0 and wa < gray.shape[1] and wa > 0):
			        frame1[wa][ha]= r_img[m][n]

	cv2.imshow("Lena", cv2.resize(frame1, (192*5,108*5)))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()


