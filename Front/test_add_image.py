import cv2


img1 = cv2.imread('1.png')
img2 = cv2.imread('red_warning.jpg')
img2  = cv2.resize(img2,(img2.shape[0]//5, img2.shape[1]//5),interpolation=cv2.INTER_AREA)

rows,cols,channels = img2.shape
print(rows,cols)
roi = img1[0:rows, 0:cols ]

print(rows, cols)

img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 230, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi,roi,mask = mask)

img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)

dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst

#cv2.imshow('img1_bg',img1_bg)
#cv2.imshow('img2_fg',img2_fg)
#cv2.imshow('dst',dst)

cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()










