import cv2
import matplotlib.pyplot as plt
#读取视频文件，读取一帧
video =cv2.VideoCapture("./")
ret,frame=video.read()
plt.imshou(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

video=cv2.VideoCapture("./")
num=0       #计数器
save_step=30#每隔30帧保存一帧
while True:
    ret,frame=video.read()
    if not ret:
        break
    num+=1
    if num%save_step==0:
        cv2.imwrite("./_images/"+str(num)+".jpg",frame)

