遇到一个需求，背景变换不大，捕捉一个发光羽毛球的运动轨迹

涉及的核心方法就是 帧间差分

看一下原始视频和最终效果

<figure class="half">
    <img src="https://oss.caibucai.top/md/opencv.gif" alt="opencv" style="zoom:30%;" />
    <img src="https://oss.caibucai.top/md/opencv2.gif" alt="opencv2" style="zoom: 30%;" />
</figure>



这个发光的就是发光羽毛球，视频的背景变化不大，还算可控
<img src="https://oss.caibucai.top/md/image-20240919174451563.png" alt="image-20240919174451563" style="zoom: 33%;" />



>  感谢 https://uutool.cn/gif-edit/  https://www.freeconvert.com/zh/gif-compressor 提供的降帧 压缩 服务



# 帧间差分

```python
frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
frame_gauss = cv.GaussianBlur(frame_gray,(5,5),0)

if background is None:
	background = frame_gauss
    continue

# diff = cv.absdiff(background,frame_gauss)

diff = cv.subtract(frame_gauss,background)

ret,thresh = cv.threshold(diff,120,255,cv.THRESH_BINARY)
# dilated = cv.dilate(thresh,es,iterations=3)
background = frame_gauss
```

打开视频流

对得到到的帧进行 灰度化、gauss 模糊，一定程度上降低 可能存在的轻微抖动，减少噪声

如果背景是绝对静止不对，background 直接使用第一帧就行，

也就不需要更新了



进行差分

```python
# diff = cv.absdiff(background,frame_gauss)

diff = cv.subtract(frame_gauss,background)

ret,thresh = cv.threshold(diff,120,255,cv.THRESH_BINARY)
```

这里测试 如果使用 absdiff() 接口 会出现 两个 目标的情况，使用 subtract() 不会出现，效果如下 右下



然后应用阈值得到二值图像，右上

![image-20240919180653120](https://oss.caibucai.top/md/image-20240919180653120.png)



然后找 边界，根据面积范围筛选到唯一的运动目标

```python
contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for c in contours:
    if cv.contourArea(c) > 250: # for ball
        continue
    print("frame {} detected, update tracking point".format({frame_id}))
    (x, y, w, h) = cv.boundingRect(c) 
    tracking_point_list[frame_id] = (x+w//2, y+h//2)
```

将边界用 矩形框保住，中心作为跟踪的目标点

# 绘制、写出

后续就是进行绘制，写出

## 绘制

使用 line() circle() rectangle() 基于保存的tracking_point_list 绘制即可。

略

## 写出

```python
out = cv.VideoWriter('task1_result.avi',int(fourcc), fps, (int(w),int(h)),True)

# ... 省去 frame 绘制

out.write(frame)
```



```python
VideoWriter(filename, fourcc, fps, frameSize, isColor)
```

1. 文件路径

2. 编码器
3. fps 
4. frameSize 帧的宽高，int 类型
5. isColor 是否为彩色

### 报错： Failed to load OpenH264 library: openh264-1.8.0-win64.dll

在这里 https://github.com/cisco/openh264/releases 下载后解压 放到 python.exe 同级目录下 即可



# 完整代码

```python
import cv2 as cv 
import numpy as np
import time

video = cv.VideoCapture('task1.avi')
if(video.isOpened()):
    n_frame = video.get(cv.CAP_PROP_FRAME_COUNT)
    w = video.get(cv.CAP_PROP_FRAME_WIDTH)
    h = video.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv.CAP_PROP_FPS)
    fourcc = video.get(cv.CAP_PROP_FOURCC)
    print("frams:{} w:{} h:{}".format(n_frame,w,h))

out = cv.VideoWriter('task1_result.avi',int(fourcc), fps, (int(w),int(h)),True)

es = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
frames = []
background = None
circle_r = 1
line_thick = 1
tracking_point = None
tracking_point_list = [None for i in range(int(n_frame))]

frame_id = 0
while video.isOpened():
    ret, frame = video.read()
    if frame is None: 
        break

    frame_id += 1
    print("frame {}".format(frame_id))
    # cv.putText(frame, 'frame:{}'.format(frame_id), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    frame_gauss = cv.GaussianBlur(frame_gray,(5,5),0)

    if background is None:
        background = frame_gauss
        continue

    # diff = cv.absdiff(background,frame_gauss)

    diff = cv.subtract(frame_gauss,background)

    ret,thresh = cv.threshold(diff,120,255,cv.THRESH_BINARY)
    # dilated = cv.dilate(thresh,es,iterations=3)
    background = frame_gauss


    # contours, hierarchy = cv.findContours(diff.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
    contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv.contourArea(c) > 250: # for ball
            continue
        print("frame {} detected, update tracking point".format({frame_id}))

        (x, y, w, h) = cv.boundingRect(c) 

        tracking_point_list[frame_id] = (x+w//2, y+h//2)

    rect_l = 2
    i = 0

    while i  < len(tracking_point_list):
        if i == 0 or  tracking_point_list[i] is None:
            i += 1
            continue
        j = i + 1
        while((j < len(tracking_point_list)) and tracking_point_list[j] is None):
            j = j + 1
        if(j >= len(tracking_point_list)):
            break
        nextPoint = tracking_point_list[j]

        cv.rectangle(frame, tracking_point_list[i], (tracking_point_list[i][0]+rect_l,tracking_point_list[i][1]+rect_l), (0, 0, 255), 2)
        # cv.circle(frame, tracking_point_list[i], 2, (0, 0, 255), 1)
        cv.line(frame, tracking_point_list[i], nextPoint, (0, 0, 255), line_thick)    
        cv.rectangle(frame, nextPoint, (nextPoint[0]+rect_l,nextPoint[1]+rect_l), (0, 0, 255), 2)

        i += 1
    
    out.write(frame)

    cv.imshow('raw',frame)
    # cv.imshow('contours', frame_gauss)
    cv.imshow('diff', diff)
    cv.imshow('thresh',thresh)

    if cv.waitKey(10) & 0xFF == 27 :
        break 
    time.sleep(0.2)

video.release()
cv.destroyAllWindows()
```


