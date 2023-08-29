import cv2
img = cv2.imread('warning.jpg')

height = img.shape[0]  # 高
width = img.shape[1]     #宽
channels = img.shape[2]  #通道数
for row in range(height):
    for col in range(width):
        if img[row, col, 2] > 150 and img[row, col, 1] >150 and  img[row, col, 0] <220:
            #img[row, col, 2] = 255
            img[row, col] = [0, 0, 255]
        #img[row, col, c] = 255 - pv

cv2.imshow("reserve",img)
cv2.imwrite("red_warning.jpg", img)
cv2.waitKey()


