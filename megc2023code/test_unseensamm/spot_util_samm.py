import sim_filter
import os
import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
import math

import matplotlib.pyplot as plt
import try_emd
import xlrd
from xlrd import xldate_as_tuple
from scipy.signal import find_peaks,peak_widths
detector = dlib.get_frontal_face_detector() #获取人脸分类器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')    # 获取人脸检测器
# Dlib 检测器和预测器
font = cv2.FONT_HERSHEY_SIMPLEX
landmark0=[]
def crop_picture(img_rd,size):
    # print(img_rd.shape)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
    #   人脸数
    faces = detector(img_gray, 0)
    # 标 68 个点
    # for i in range(len(faces)):
    #     # 取特征点坐标
    #     landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[i]).parts()])

    landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[0]).parts()])
    #两个眼角的位置
    left=landmarks[39]
    right=landmarks[42]
    # print(left)
    # print(right)
    thete=math.atan(float(right[0,1]-left[0,1])/(right[0,0]-left[0,0]))
    # print("角度是{}".format(thete))

    gezi=int((right[0,0]-left[0,0])/2)
    center=[int((right[0,0]+left[0,0])/2),int((right[0,1]+left[0,1])/2)]
    # print(8*gezi)

    #从中心点阔8格
    # cv2.rectangle(img_rd, (center[0]-4*gezi, center[1]-3*gezi), (center[0]+4*gezi, center[1]+5*gezi), (0, 0, 255), 2)
    # img_crop=img_rd[center[1]-3*gezi:center[1]+5*gezi,center[0]-4*gezi:center[0]+4*gezi]
    # img_crop_samesize=cv2.resize(img_crop,(size,size))
    # return landmarks,img_crop_samesize,center[1]-3*gezi,center[1]+5*gezi,center[0]-4*gezi,center[0]+4*gezi
    # 从中心点阔10格
    # cv2.rectangle(img_rd, (center[0] - 5 * gezi, center[1] - 4 * gezi), (center[0] + 5 * gezi, center[1] + 6 * gezi),
    #               (0, 0, 255), 2)
    # img_crop = img_rd[center[1] - 4 * gezi:center[1] + 6 * gezi, center[0] - 5 * gezi:center[0] + 5 * gezi]
    # img_crop_samesize = cv2.resize(img_crop, (size, size))
    # return landmarks, img_crop_samesize, center[1] - 4* gezi, center[1] + 6 * gezi, center[0] - 5 * gezi, center[
    #     0] + 5 * gezi
    # 从中心点阔9格
    cv2.rectangle(img_rd, (center[0] - int(4.5 * gezi), center[1] - int(3.5 * gezi)), (center[0] + int(4.5 * gezi), center[1] + int(5.5 * gezi)),
                  (0, 0, 255), 2)

    a=(center[1] - int(3.5 * gezi))
    b=center[1] +int(5.5 * gezi)
    c=(center[0] - int(4.5 * gezi))
    d=center[0] +int(4.5 * gezi)

    a=max((center[1] - int(3.5 * gezi)),0)
    # b=min(center[1] +int(5.5 * gezi),399)
    c=max(center[0] - int(4.5 * gezi),0)
    # d=min(center[0] +int(4.5 * gezi),399)
    img_crop = img_rd[a:b, c:d]
    # print(img_crop.shape)
    # cv2.imshow("image", img_crop)
    # cv2.waitKey(0)
    img_crop_samesize = cv2.resize(img_crop, (size, size))
    return landmarks, img_crop_samesize, a, b, c, d
    # 窗口显示
    # 参数取 0 可以拖动缩放窗口，为 1 不可以
    # cv2.namedWindow("image", 0)
    # cv2.namedWindow("image", 1)
    # cv2.imshow("image", img_crop_samesize)
    # cv2.waitKey(0)
    # cv2.imshow("image", img_rd)
    # cv2.waitKey(0)

def get_roi_bound(low,high,round,landmark0):

    roi1_points = landmark0[low:high]
    #print(roi1_points)

    roi1_high = roi1_points[:, 0].argmax(axis=0)
    roi1_low = roi1_points[:, 0].argmin(axis=0)
    roi1_left = roi1_points[:, 1].argmin(axis=0)
    roil_right = roi1_points[:, 1].argmax(axis=0)

    roil_h = roi1_points[roi1_high, 0]
    roi1_lo = roi1_points[roi1_low, 0]
    roi1_le = roi1_points[roi1_left, 1]
    roil_r = roi1_points[roil_right, 1]

    roil_h_ex = (roil_h + round)[0, 0]
    roi1_lo_ex = (roi1_lo - round)[0, 0]
    roi1_le_ex = (roi1_le - round)[0, 0]
    roil_r_ex = (roil_r + round)[0, 0]
    return (roil_h_ex),(roi1_lo_ex),(roi1_le_ex),(roil_r_ex)
def get_roi(flow, percent):
    # 光流的直角坐标系
    r1, theta1 = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
    r1 = np.ravel(r1)
    # 光流拉平成一维
    x1 = np.ravel(flow[:, :, 0])
    y1 = np.ravel(flow[:, :, 1])

    arg = np.argsort(r1)  # 代表了r1这个矩阵内元素的从小到大顺序
    num = int(len(r1) * (1 - percent))
    x_new = 0
    y_new = 0
    # 取后70%的强度的光流
    #print("len arg:{},len x1:{}".format(len(arg),len(x1)))
    for i in range(num, len(arg)):  # 想取相对比较大的
        a = arg[i]
        x_new += x1[a]
        y_new += y1[a]
    x = x_new / (len(arg) - num)
    y = y_new / (len(arg) - num)
    return x, y

# 返回图像的68个标定点
# def tu_landmarks(gray, img_rd, landmark0, frame_shang, frame_left, w, h, img_size):
#w h反了
def tu_landmarks(gray, img_rd, landmark0, frame_shang, frame_left, h, w, img_size):
    faces = detector(gray, 0)
    if (len(faces) == 0):#就是原图根据关键点Crop  如果crop之后的检测不到关键点  就使用原图的关键点按比例缩放一下
        landmark0[:, 0] = (landmark0[:, 0] - frame_left) * (img_size / w)
        landmark0[:, 1] = (landmark0[:, 1] - frame_shang) * (img_size / h)
        landmarkss = landmark0
    else:
        landmarkss = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[0]).parts()])
    return landmarkss

#对给定的每个视频帧之间的光流。进行求平方和和开根号的计算，并画出动作线
def draw_line(flow_total):
    flow_total=np.array(flow_total)

    flow_total=np.sum(flow_total**2,axis=1)
    flow_total=np.sqrt(flow_total)

    return flow_total
def fenxi(flow_total, imf_sum1,yuzhi1,yuzhi2):
    flow_total = np.array(flow_total)
    low=np.min(flow_total)
    flow_total=flow_total-low
    flow_total_fenxi=[]
    for j in range(len(flow_total)):
        if(flow_total[j]>=yuzhi1):
            flow_total_fenxi.append(j)
    flow_total_pp = []
    if(len(flow_total_fenxi)>0):
        start=flow_total_fenxi[0]
        end=flow_total_fenxi[0]
        st=0
        for i in range(len(flow_total_fenxi)):
            if(flow_total_fenxi[i]>=end and flow_total_fenxi[i]-end<5):
                end=flow_total_fenxi[i]
            else:
                flow_total_pp.append([start,end])
                start = flow_total_fenxi[i]
                end = flow_total_fenxi[i]
        flow_total_pp.append([start, end])
    flow_total_fenxi = []
    flow_total_pp=np.array(flow_total_pp)
    for i in range(len(flow_total_pp)):
        start=flow_total_pp[i,0]
        end = flow_total_pp[i, 1]
        # start_new=min(0,start-25)
        # end_new=max(len(flow_total-1),end+25)
        # low = np.min(flow_total[start_new:end_new])
        # flow_total[start:end]=flow_total[start:end]-low
        for j in range(start,end):
            a = max(0, j - 30)
            b=min(len(flow_total-1),j+30)
            low=np.min(flow_total[a:b])

            if(flow_total[j]-low>yuzhi2 ):
                flow_total_fenxi.append(j)
    flow_total_pp2 = []
    if (len(flow_total_fenxi) > 0):
        start = flow_total_fenxi[0]
        end = flow_total_fenxi[0]
        st = 0
        for i in range(len(flow_total_fenxi)):
            if (flow_total_fenxi[i] >= end and flow_total_fenxi[i] - end < 5):
                end = flow_total_fenxi[i]
            else:
                flow_total_pp2.append([start, end])
                start = flow_total_fenxi[i]
                end = flow_total_fenxi[i]
        flow_total_pp2.append([start, end])

    return np.array(flow_total_pp2)
def fenxi1(flow_total,imf_sum1,yuzhi1,yuzhi2): #使用寻找峰的方法
    flow_total = np.array(flow_total)
    low=np.min(flow_total)    #找到最小值
    flow_total=flow_total-low     #从零开始
    flow_total_fenxi=[]
    for j in range(len(flow_total)):#找到大于较小阈值
        if(flow_total[j]>=yuzhi1):
            flow_total_fenxi.append(j)   #大于较小阈值的帧的索引
    flow_total_pp = []
    if(len(flow_total_fenxi)>0): #对经过第一步筛选的，帧相邻的连在一起
        start=flow_total_fenxi[0]
        end=flow_total_fenxi[0]
        st=0
        for i in range(len(flow_total_fenxi)):
            if(flow_total_fenxi[i]>=end and flow_total_fenxi[i]-end<3):
                end=flow_total_fenxi[i]
            else:
                flow_total_pp.append([start,end])
                start = flow_total_fenxi[i]
                end = flow_total_fenxi[i]
        flow_total_pp.append([start, end])
    flow_total_fenxi = []
    flow_total_pp=np.array(flow_total_pp)

    for i in range(len(flow_total_pp)): #第二次筛选
        start=flow_total_pp[i,0]
        end = flow_total_pp[i, 1]
        # start_new=min(0,start-25)
        # end_new=max(len(flow_total-1),end+25)
        # low = np.min(flow_total[start_new:end_new])
        # flow_total[start:end]=flow_total[start:end]-low
        for j in range(start,end):
            a = max(0, j - 30)
            b  = min(len(flow_total)-1,j+30)  #找到这个点的两边，左边右边各30，注意不能超过滑动窗口的碧娜姐
            low= np.min(flow_total[a:b])     #左右区间都找最小的
            low1 = np.min(imf_sum1[a:b])     #左右区间都找最小的
            if (flow_total[j] - low > yuzhi2 and imf_sum1[j] - low1 >0.8):
                flow_total_fenxi.append(j)
    flow_total_pp2 = []
    if (len(flow_total_fenxi) > 0):
        start = flow_total_fenxi[0]
        end = flow_total_fenxi[0]
        st = 0
        for i in range(len(flow_total_fenxi)):
            if (flow_total_fenxi[i] >= end and flow_total_fenxi[i] - end < 3):
                end = flow_total_fenxi[i]
            else:
                flow_total_pp2.append([start, end])
                start = flow_total_fenxi[i]
                end = flow_total_fenxi[i]
        # start=max(start-5,0)
        # end=min(end+5,199)
        flow_total_pp2.append([start, end])

    return np.array(flow_total_pp2)

def expend(flow1_total_fenxi,flow1_total_edm):
    for i in range(len(flow1_total_fenxi)):
        start=flow1_total_fenxi[i, 0]
        end=flow1_total_fenxi[i,1]
        a1 = max(0, start - 30)
        b1 = min(len(flow1_total_edm )-1, start + 30)
        a2 = max(0,  end- 30)
        b2 = min(len(flow1_total_edm) - 1, end + 30)
        if(end>start):  #因为有可能end=start
            high=np.max(flow1_total_edm[start:end])
        else:
            high=flow1_total_edm[start]

        st_low = np.min(flow1_total_edm[a1:b1])
        st_arglow=np.argmin(flow1_total_edm[a1:b1])+a1  #start的左右中最小的索引
        en_low = np.min(flow1_total_edm[a2:b2])      #end的左右中最小的索引
        en_arglow=np.argmin(flow1_total_edm[a2:b2])+a2
        if(st_arglow<start):
            for j in range(start-1,-1,-1):
                if(flow1_total_edm[j]-st_low<0.33*(high-st_low)):
                    start=j
                    break
                if (flow1_total_edm[j] > flow1_total_edm[j + 1]):
                    start = j + 2
                    break
        else:
            left=max(start-10,0)
            aa=np.argmin(flow1_total_edm[left:start+1]) +left #代表了start左侧十个中值最小的索引
            if(flow1_total_edm[start]-flow1_total_edm[aa]>0.3):
                start=aa+1
        if (en_arglow > end):
            for j in range(end+1, en_arglow):
                if (flow1_total_edm[j] - en_low < 0.33*(high-en_low)):
                    end = j
                    break
                if(flow1_total_edm[j]>flow1_total_edm[j-1]):
                    end=j-2
                    break
        else:
            right=min(end+10,len(flow1_total_edm )-1)
            aa=np.argmin(flow1_total_edm[end:right+1])+end#代表了end右侧十个中值最小的索引
            if(flow1_total_edm[end]-flow1_total_edm[aa]>0.3):
                end=aa-1  #用最小值的索引进行替换

        flow1_total_fenxi[i, 0]=start
        flow1_total_fenxi[i, 1]=end
    return flow1_total_fenxi

def divide(flow1_total_fenxi, flow1_total_edm):
    h=[]
    for i in range(len(flow1_total_fenxi)):
        a=int(flow1_total_fenxi[i,0])
        b=int(flow1_total_fenxi[i,1])
        if((flow1_total_fenxi[i,1]-flow1_total_fenxi[i,0])>=20):
            minnum=np.argmin(flow1_total_edm[a+8:b-8])+a+8
            max1=np.max(flow1_total_edm[a:minnum])
            max2=np.max(flow1_total_edm[minnum:b])
            if((max1-flow1_total_edm[minnum]>max(0.7,0.33*max1)) and (max2-flow1_total_edm[minnum]>max(0.7,0.33*max2))):
                h.append([a,minnum-1])
                h.append([minnum+1,b])
            else:
                h.append([a,b])
        else:
            h.append([a,b])
    return np.array(h)

def proce2(flow1_total,yuzhi1,yuzhi2,position,xuhao,k,a,totalflow,totalflow_mic,totalflow_mac):
    fs=1
    c1=-0.2
    yuzhi1=yuzhi1+c1
    c2=-0.1
    yuzhi2=yuzhi2+c2
    flow1_total = draw_line(flow1_total)#作用是将光流特征转换为幅值的形式
    flow1_total = np.array(flow1_total)
    position=position+str(xuhao)+"----"  #
    flow1_total_edm1 = sim_filter.filt(flow1_total[a:-a], 1, 5, 30)  # 滤波
    #flow1_total_edm1 = sim_filter.filt(flow1_total[a:-a], 1, 3, 30)
    hh = len(flow1_total_edm1)+2 #作用等同于200

    flow1_total_edm2,imf_sum1 = try_emd.the_emd1(flow1_total[a:-a],flow1_total_edm1, position, str(k - hh) ,fs)

    flow1_total_fenxi = fenxi1(flow1_total_edm1,imf_sum1,yuzhi1,yuzhi2)  #得到了分析结果
    flow1_total_fenxi=expend(flow1_total_fenxi,flow1_total_edm1)   #向两边扩展
    flow1_total_fenxi = divide(flow1_total_fenxi, flow1_total_edm1)#将中间低的峰分成两个部分。
    flow1_total_fenxi=flow1_total_fenxi+ (k - hh) + a
    for i in range(len(flow1_total_fenxi)):
        totalflow.append(flow1_total_fenxi[i])


    return totalflow,totalflow_mic,totalflow_mac
def nms2(totalflow):
    totalflow=np.array(totalflow)
    hh=[[0,0]]
    for i in range(len(totalflow)):
        new=1
        if(i==0):
            hh=np.vstack((hh,[[totalflow[i,0],totalflow[i,1]]]))
            continue
        for j in range(1,len(hh)):
            # print("pp")
            # print(len(hh))
            #计算iou
            if(totalflow[i,0]>hh[j,1] or totalflow[i,1]<hh[j,0]):   #两个间隔完全不相交
                iou=0
            else:
                ma=max(totalflow[i,0],hh[j,0])
                mi=min(totalflow[i,1],hh[j,1])
                wid=mi-ma
                if wid == 0:
                    iou = 0
                else:
                    iou=max(wid/(hh[j,1]-hh[j,0]),wid/(totalflow[i,1]-totalflow[i,0]))
            #通过iou决定是不是要添加
            # if(iou>0.34): #SAMM0.34  CASME 0.29   #如果重复率比较高就
            # if (iou > 0.25):
            if (iou > 0.2):
            #if (iou > 0.15):
                new=0
                hh[j, 1]=max(hh[j, 1],totalflow[i, 1])
                hh[j, 0]=min(hh[j, 0],totalflow[i, 0])
        if(new ==1):
            hh=np.vstack((hh, [[totalflow[i, 0], totalflow[i, 1]]]))
    return hh


def draw_roiline19(path1,path2, qian, hou,fs):  #与16相比再增加两个位置眼睑部位
    #folder = os.path.exists("C:/sheng/SAMM/test_SAMM_crop5/"+path2)
    # pathp = "C:/sheng/SAMM/test_SAMM_crop5/" + path2   #存储的位置
    # print(pathp)
    # if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
    #     os.makedirs(pathp)

    path=path1+'/'+path2+'/'    #视频图片文件夹的位置
    fileList1 = os.listdir(path)  #图片路径
    #fileList1.sort(key=lambda x: int(x[qian:hou]))  #对提取的图片排序
    fileList1.sort(key=lambda x: int(x.split('_')[2][0:-4]))  #no openface
    #fileList1.sort(key=lambda x: int(x.split('_')[3][0:-4]))

    fileList = []
    l=0
    for i in fileList1:
        if(l%fs==0):
            fileList.append(i)
        l = l + 1
    #print(fileList)
    k =0#这里的k代表开始的位置
    start = k-99 #每一小段的开始和结束
    end = k+100
    move=100    #默认移动是100
    last=True   #最后一段是否处理过
    lable_vio=np.array([0,0])
    while(k<len(fileList)):
    # while(k<len(fileList) and k<2000):
        start+=move
        end+=move
        if(end>len(fileList) and last==True):

            end =len(fileList)-2  #如果是最后一个，及没有200那么多，就调整end，   start不变
            last=False
        k=0
        mid=False
        for i in fileList:
            k = k + 1
            if(k>=start):
                if(k==start):
                    flow1_total = [[0, 0]]  # 是存储了不同位置帧之间的光流
                    flow1_total1 = [[0, 0]]
                    flow1_total2 = [[0, 0]]
                    flow1_total3 = [[0, 0]]
                    flow2_total = [[0, 0]]
                    flow3_total = [[0, 0]]
                    flow3_total1 = [[0, 0]]
                    flow3_total2 = [[0, 0]]
                    flow3_total3 = [[0, 0]]
                    flow4_total = [[0, 0]]
                    flow4_total1 = [[0, 0]]
                    flow4_total2 = [[0, 0]]
                    flow4_total3 = [[0, 0]]
                    flow4_total4 = [[0, 0]]
                    flow4_total5 = [[0, 0]]
                    flow5_total1 = [[0, 0]]
                    flow5_total2 = [[0, 0]]
                    flow2_total1 = [[0, 0]]
                    flow6_total= [[0, 0]]
                    flow7_total= [[0, 0]]

                   # print("k:{},start :{},im:{}".format(k,start,path + i))
                    img_rd = cv2.imread(path + i)  # D:/face_image_test/EP07_04/
                    landmark0, img_rd, frame_shang, frame_xia, frame_left, frame_right = crop_picture(img_rd,320)
                    #记录框的位置，上下左右在整个图片中的坐标，和68点的位置。img_rd是被裁减之后的面部位置，并resize到320*320


                    gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)  #变成灰度图
                    cv2.imwrite("./firstframe_cropgray.jpg", gray)
                    img_size=320
                    #landmark0 = tu_landmarks(gray, img_rd, landmark0, frame_shang, frame_left)  # 对人脸68个点的定位
                    landmark0 = tu_landmarks(gray, img_rd, landmark0, frame_shang, frame_left, frame_xia - frame_shang,
                                             frame_right - frame_left, img_size)
                    #相对与新图片的68点的位置。

                    round1 = 0
                    roil_right, roi1_left, roi1_low, roi1_high = get_roi_bound(17, 22, 0, landmark0)   #左眉毛的位置
                    # cv2.rectangle(img_rd, (roi1_left-5, roi1_low - 15), (roil_right, roi1_high + 5), (0, 255, 0), 1)

                    roi1_sma = []  # 存储了左眼的三个小的感兴趣区域，从里到外
                    roi1_sma.append([landmark0[20, 1] - (roi1_low - 15), landmark0[20, 0] - (roi1_left-5)])
                    roi1_sma.append([landmark0[19, 1] - (roi1_low - 15), landmark0[19, 0] - (roi1_left-5)])
                    roi1_sma.append([landmark0[18, 1] - (roi1_low - 15), landmark0[18, 0] - (roi1_left-5)])
                    cv2.rectangle(img_rd, (landmark0[20, 0] - 10, landmark0[20, 1] + 10),
                                  (landmark0[20, 0] + 10, landmark0[20, 1] - 10), (0, 255, 255), 1)
                    cv2.rectangle(img_rd, (landmark0[19, 0] - 10, landmark0[19, 1] + 10),
                                  (landmark0[19, 0] + 10, landmark0[19, 1] - 10), (0, 255, 255), 1)
                    cv2.rectangle(img_rd, (landmark0[18, 0] - 10, landmark0[18, 1] + 10),
                                  (landmark0[18, 0] + 10, landmark0[18, 1] - 10), (0, 255, 255), 1)
                    # prevgray_roi1 = gray[(roi1_low - 15):roi1_high + 5, roi1_left-5:roil_right]
                    prevgray_roi1 = gray[max(0, (roi1_low - 15)):roi1_high + 5, roi1_left - 5:roil_right]

                    # 右眼
                    roi3_right, roi3_left, roi3_low, roi3_high = get_roi_bound(22, 27, 0, landmark0)
                    # cv2.rectangle(img_rd, (roi3_left, roi3_high + 5), (roi3_right, roi3_low - 15), (0, 255, 0), 1)
                    roi3_sma = []  # 存储了右眼的三个小的感兴趣区域，从里到外
                    roi3_sma.append([landmark0[23, 1] - (roi3_low - 15), landmark0[23, 0] - roi3_left])
                    roi3_sma.append([landmark0[24, 1] - (roi3_low - 15), landmark0[24, 0] - roi3_left])
                    roi3_sma.append([landmark0[25, 1] - (roi3_low - 15), landmark0[25, 0] - roi3_left])
                    cv2.rectangle(img_rd, (landmark0[25, 0] - 10, landmark0[25, 1] + 10),
                                  (landmark0[25, 0] + 10, landmark0[25, 1] - 10), (0, 255, 255), 1)
                    cv2.rectangle(img_rd, (landmark0[24, 0] - 10, landmark0[24, 1] + 10),
                                  (landmark0[24, 0] + 10, landmark0[24, 1] - 10), (0, 255, 255), 1)
                    cv2.rectangle(img_rd, (landmark0[23, 0] - 10, landmark0[23, 1] + 10),
                                  (landmark0[23, 0] + 10, landmark0[23, 1] - 10), (0, 255, 255), 1)
                   # prevgray_roi3 = gray[(roi3_low - 15):roi3_high + 5, roi3_left:roi3_right]
                    prevgray_roi3 = gray[max(0, (roi3_low - 15)):roi3_high + 5, roi3_left:roi3_right]
                    # print(prevgray_roi1.shape)

                    # 嘴巴处的四个
                    roi4_right, roi4_left, roi4_low, roi4_high = get_roi_bound(48, 67, 0, landmark0)
                    # cv2.rectangle(img_rd, (roi4_left-20, roi4_high + 10), (roi4_right+20, roi4_low - 15), (0, 255, 0), 1)
                    roi4_sma = []
                    roi4_sma.append([landmark0[48, 1] - (roi4_low - 15), landmark0[48, 0] - (roi4_left - 20)])
                    roi4_sma.append([landmark0[54, 1] - (roi4_low - 15), landmark0[54, 0] - (roi4_left - 20)])
                    roi4_sma.append([landmark0[51, 1] - (roi4_low - 15), landmark0[51, 0] - (roi4_left - 20)])
                    roi4_sma.append([landmark0[57, 1] - (roi4_low - 15), landmark0[57, 0] - (roi4_left - 20)])
                    roi4_sma.append([landmark0[62, 1] - (roi4_low - 15), landmark0[62, 0] - (roi4_left - 20)])
                    cv2.rectangle(img_rd, (landmark0[48, 0] -10, landmark0[48, 1] + 10),
                                  (landmark0[48, 0] + 10, landmark0[48, 1] - 10), (0, 255, 255), 1)
                    cv2.rectangle(img_rd, (landmark0[51, 0] - 10, landmark0[51, 1] + 10),
                                  (landmark0[51, 0] + 10, landmark0[51, 1] - 10), (0, 255, 255), 1)
                    cv2.rectangle(img_rd, (landmark0[54, 0] - 10, landmark0[54, 1] + 10),
                                  (landmark0[54, 0] + 10, landmark0[54, 1] - 10), (0, 255, 255), 1)
                    cv2.rectangle(img_rd, (landmark0[57, 0] - 10, landmark0[57, 1] + 10),
                                  (landmark0[57, 0] + 10, landmark0[57, 1] - 10), (0, 255, 255), 1)
                    cv2.rectangle(img_rd, (landmark0[62, 0] - 10, landmark0[62, 1] + 10),
                                  (landmark0[62, 0] + 10, landmark0[62, 1] - 10), (0, 255, 0), 1)
                    prevgray_roi4 = gray[(roi4_low - 15):roi4_high + 10, roi4_left - 20:roi4_right + 20]

                    # 鼻子两侧
                    roi5_right, roi5_left, roi5_low, roi5_high = get_roi_bound(30, 36, 0, landmark0)
                    # cv2.rectangle(img_rd, (roi5_left-22, roi5_high + 5), (roi5_right+22, roi5_low - 20), (0, 255, 0), 1)
                    roi5_sma = []
                    roi5_sma.append([landmark0[31, 1] - (roi5_low - 20), landmark0[31, 0] - (roi5_left - 30)])
                    roi5_sma.append([landmark0[35, 1] - (roi5_low - 20), landmark0[35, 0] - (roi5_left - 30)])

                    cv2.rectangle(img_rd, (landmark0[31, 0] -20, landmark0[31, 1] + 5),
                                  (landmark0[31, 0] + 10, landmark0[31, 1] - 20), (0, 255, 255), 1)
                    cv2.rectangle(img_rd, (landmark0[35, 0] - 10, landmark0[35, 1] + 5),
                                  (landmark0[35, 0] + 20, landmark0[35, 1] - 20), (0, 255, 255), 1)

                    prevgray_roi5 = gray[(roi5_low - 20):roi5_high + 5, roi5_left - 30:roi5_right + 30]

                    #左眼睑部位
                    roi6_right, roi6_left, roi6_low, roi6_high = get_roi_bound(36, 42, 0, landmark0)
                    # print(roi6_high)//100
                    # print(roi6_low)//90
                    width=roi6_right-roi6_left
                    height=width/2
                    xin=(roi6_high+ roi6_low)/2
                    roi6_high=int(xin+3*height/2)
                    roi6_low=int(xin+height/2)
                    prevgray_roi6 = gray[roi6_low :roi6_high, roi6_left :roi6_right]
                    #cv2.rectangle(img_rd, (roi6_left, roi6_high ), (roi6_right, roi6_low ), (0, 255, 255), 1)

                    # 右眼睑部位
                    roi7_right, roi7_left, roi7_low, roi7_high = get_roi_bound(42, 48, 0, landmark0)
                    # print(roi6_high)//100
                    # print(roi6_low)//90
                    width = roi7_right - roi7_left
                    height = width / 2
                    xin = (roi7_high + roi7_low) / 2
                    roi7_high = int(xin + 3 * height / 2)
                    roi7_low = int(xin + height / 2)
                    prevgray_roi7 = gray[roi7_low :roi7_high , roi7_left :roi7_right ]
                    #cv2.rectangle(img_rd, (roi7_left, roi7_high), (roi7_right, roi7_low), (0, 255, 255), 1)




                    roi2_right, roi2_left, roi2_low, roi2_high = get_roi_bound(29, 31, 13, landmark0)
                    prevgray_roi2 = gray[roi2_low:roi2_high, roi2_left:roi2_right]

                    # cv2.rectangle(img_rd, (roi2_left + round1, roi2_high + 5 - round1),
                    #               (roi2_right - round1, roi2_low - 10 + round1), (0, 255, 0), 1)
                    # cv2.rectangle(img_rd, (roi2_left + 5, roi2_high + 5 - 15), (roi2_right - 5, roi2_low - 10 + 25),
                    #               (0, 255, 0), 1)
                    # path33 = "D:/dataset/micro_datatset/imageforpaper/img2.jpg"
                    cv2.imwrite('./landmarkroi.jpg', img_rd)
                    # cv2.imshow("image1", img_rd)
                    # cv2.waitKey(0)
                    # print("ll")
                    # print(prevgray_roi2.shape)


                else:

                    if (True):
                       # print(path+i)

                        img_rd1 = cv2.imread(path + i)  # D:/face_image_test/EP07_04/
                        img_crop = img_rd1[frame_shang:frame_xia, frame_left:frame_right]  # 按照第一个图的框切割出一个脸

                        img_rd = cv2.resize(img_crop, (320, 320))
                        gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
                        # 求全局的光流
                        gray_roi2 = gray[roi2_low:roi2_high, roi2_left:roi2_right]
                        cv2.imwrite("./grayroi2.jpg", gray_roi2)
                        # 使用Gunnar Farneback算法计算密集光流
                        flow2 = cv2.calcOpticalFlowFarneback(prevgray_roi2, gray_roi2, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                        flow2 = np.array(flow2)

                        # him2, x1, y1 = get_roi_him(flow2[15:-10, 5:-5, :])
                        x1, y1 = get_roi(flow2[15:-10, 5:-5, :], 0.7)
                        # print("全局运动为{}and{}".format(x1,y1))
                        flow2_total1.append([x1, y1])

                        #进行面部对齐，移动切割框
                        l = 0
                        while ((x1 ** 2 + y1 ** 2) > 1):  # 移动比较大，相应移动脸的位置
                            l = l + 1
                            if (l > 3):
                                print("ppp")
                                break
                            frame_left += int(round(x1))
                            frame_shang += int(round(y1))
                            frame_right += int(round(x1))
                            frame_xia += int(round(y1))
                            # print(frame_left)
                            # print(frame_shang)
                            # print(frame_right)
                            # print(frame_xia)
                            img_rd1 = cv2.imread(path + i)
                            img_crop = img_rd1[frame_shang:frame_xia, frame_left:frame_right]
                            img_rd = cv2.resize(img_crop, (320, 320))
                            gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
                            # 求全局的光流
                            gray_roi2 = gray[roi2_low:roi2_high, roi2_left:roi2_right]
                            # 使用Gunnar Farneback算法计算密集光流
                            flow2 = cv2.calcOpticalFlowFarneback(prevgray_roi2, gray_roi2, None, 0.5, 3, 15, 5, 7, 1.5,
                                                                 0)
                            flow2 = np.array(flow2)

                            # him2, x1, y1 = get_roi_him(flow2[15:-10, 5:-5, :])
                            x1, y1 = get_roi(flow2[15:-10, 5:-5, :], 0.7)

                            # print("全局运动为{}and{}".format(x1, y1))
                            flow2_total1.append([x1, y1])
                        #对齐完毕
                        cv2.imwrite("./cropalign_gray.jpg", gray)

                        #gray_roi1 = gray[(roi1_low - 15):roi1_high + 5, roi1_left-5:roil_right]
                        gray_roi1 = gray[max(0, (roi1_low - 15)):roi1_high + 5, roi1_left - 5:roil_right]
                        cv2.imwrite("./grayroi1.jpg", gray_roi1)
                        # print("low:{},high:{},lef:{},right:{}".format(roi1_low,roi1_high,roi1_left,roil_right))
                        # # 使用Gunnar Farneback算法计算密集光流
                        # print(prevgray_roi1.shape)
                        # print(gray_roi1.shape)

                        flow1 = cv2.calcOpticalFlowFarneback(prevgray_roi1, gray_roi1, None, 0.5, 3, 15, 5, 7, 1.5, 0)#计算整个左眉毛处的光流
                        flow1[:, :, 0] = flow1[:, :, 0]
                        flow1[:, :, 1] = flow1[:, :, 1]
                        # print("pppppp")
                        round1 = 10
                        roi1_sma = np.array(roi1_sma)
                        # print(roi1_sma)
                        a, b = get_roi(flow1[round1:-round1, round1:-round1, :], 0.2)   #去掉光流特征矩阵周边round大小的部分，求均值
                        a1, b1 = get_roi(           #一个感兴趣区域处的平均光流
                            flow1[roi1_sma[0, 0] - 10:roi1_sma[0, 0] + 10, roi1_sma[0, 1] - 10:roi1_sma[0, 1] + 10, :],
                            0.2)
                        a2, b2 = get_roi(
                            flow1[roi1_sma[1, 0] - 10:roi1_sma[1, 0] + 10, roi1_sma[1, 1] - 10:roi1_sma[1, 1] + 10, :],
                            0.2)
                        a3, b3 = get_roi(
                            flow1[roi1_sma[2, 0] - 10:roi1_sma[2, 0] + 10, roi1_sma[2, 1] - 10:roi1_sma[2, 1] + 10, :],
                            0.2)

                        flow1_total1.append([a1 - x1, b1 - y1])  #局部区域减去全局光流
                        flow1_total2.append([a2 - x1, b2 - y1])
                        flow1_total3.append([a3 - x1, b3 - y1])
                        flow1_total.append([a - x1, b - y1])

                       # gray_roi3 = gray[(roi3_low - 15):roi3_high + 5, roi3_left:roi3_right]
                        gray_roi3 = gray[max(0, (roi3_low - 15)):roi3_high + 5, roi3_left:roi3_right]
                        cv2.imwrite("./grayroi3.jpg", gray_roi3)
                        # 使用Gunnar Farneback算法计算密集光流
                        flow3 = cv2.calcOpticalFlowFarneback(prevgray_roi3, gray_roi3, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                        flow3[:, :, 0] = flow3[:, :, 0]
                        flow3[:, :, 1] = flow3[:, :, 1]
                        round1 = 10
                        # a = np.mean(flow3[round1:-round1, round1:-round1, 0])
                        # b = np.mean(flow3[round1:-round1, round1:-round1, 1])
                        roi3_sma = np.array(roi3_sma)
                        # print(roi1_sma)
                        a, b = get_roi(flow3[round1:-round1, round1:-round1, :], 0.3)
                        a1, b1 = get_roi(
                            flow3[roi3_sma[0, 0] - 10:roi3_sma[0, 0] + 10, roi3_sma[0, 1] - 10:roi3_sma[0, 1] + 10, :],
                            0.3)
                        a2, b2 = get_roi(
                            flow3[roi3_sma[1, 0] - 10:roi3_sma[1, 0] + 10, roi3_sma[1, 1] - 10:roi3_sma[1, 1] + 10, :],
                            0.3)
                        a3, b3 = get_roi(
                            flow3[roi3_sma[2, 0] - 10:roi3_sma[2, 0] + 10, roi3_sma[2, 1] - 10:roi3_sma[2, 1] + 10, :],
                            0.3)

                        flow3_total1.append([a1 - x1, b1 - y1])
                        flow3_total2.append([a2 - x1, b2 - y1])
                        flow3_total3.append([a3 - x1, b3 - y1])
                        flow3_total.append([a - x1, b - y1])

                        gray_roi4 = gray[(roi4_low - 15):roi4_high + 10, roi4_left - 20:roi4_right + 20]
                        cv2.imwrite("./grayroi4.jpg", gray_roi4)
                        # print(gray_roi4.shape)
                        # print(prevgray_roi4.shape)
                        # 使用Gunnar Farneback算法计算密集光流
                        flow4 = cv2.calcOpticalFlowFarneback(prevgray_roi4, gray_roi4, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                        flow4[:, :, 0] = flow4[:, :, 0]
                        flow4[:, :, 1] = flow4[:, :, 1]
                        round1 = 10
                        roi4_sma = np.array(roi4_sma)
                        # print(roi1_sma)
                        a, b = get_roi(flow4[round1:-round1, round1:-round1, :], 0.3)
                        a1, b1 = get_roi(
                            flow4[roi4_sma[0, 0] - 10:roi4_sma[0, 0] + 10, roi4_sma[0, 1] - 10:roi4_sma[0, 1] + 20, :],
                            0.2)
                        a2, b2 = get_roi(
                            flow4[roi4_sma[1, 0] - 10:roi4_sma[1, 0] + 10, roi4_sma[1, 1] - 20:roi4_sma[1, 1] + 10, :],
                            0.2)
                        a3, b3 = get_roi(
                            flow4[roi4_sma[2, 0] - 10:roi4_sma[2, 0] + 10, roi4_sma[2, 1] - 10:roi4_sma[2, 1] + 10, :],
                            0.2)
                        a4, b4 = get_roi(
                            flow4[roi4_sma[3, 0] - 10:roi4_sma[3, 0] + 10, roi4_sma[3, 1] - 10:roi4_sma[3, 1] + 10, :],
                            0.2)
                        a5, b5 = get_roi(
                            flow4[roi4_sma[4, 0] - 10:roi4_sma[4, 0] + 10, roi4_sma[4, 1] - 10:roi4_sma[4, 1] + 10, :],
                            0.2)

                        flow4_total1.append([a1 - x1, b1 - y1])
                        flow4_total2.append([a2 - x1, b2 - y1])
                        flow4_total3.append([a3 - x1, b3 - y1])
                        flow4_total4.append([a4 - x1, b4 - y1])
                        flow4_total5.append([a5 - x1, b5 - y1])
                        flow4_total.append([a - x1, b - y1])

                        gray_roi5 = gray[(roi5_low - 20):roi5_high + 5, roi5_left - 30:roi5_right + 30]
                        cv2.imwrite("./grayroi5.jpg", gray_roi5)
                        # 使用Gunnar Farneback算法计算密集光流
                        flow5 = cv2.calcOpticalFlowFarneback(prevgray_roi5, gray_roi5, None, 0.5, 3, 15, 5, 7, 1.5, 0)

                        round1 = 10
                        roi5_sma = np.array(roi5_sma)
                        # print(roi1_sma)
                        # print("=========")
                        # print(roi5_sma)
                        # print(flow5.shape)
                        # print(roi5_sma)
                        #
                        # print(flow5[roi5_sma[0, 0] - 25:roi5_sma[0, 0] + 5, roi5_sma[0, 1] - 20:roi5_sma[0, 1] + 10,
                        #       :].shape)
                        # print(flow5[roi5_sma[1, 0] - 25:roi5_sma[1, 0] + 5, roi5_sma[1, 1] - 20:roi5_sma[1, 1] + 10,
                        #       :].shape)
                        a1, b1 = get_roi(
                            flow5[roi5_sma[0, 0] - 20:roi5_sma[0, 0] + 5, roi5_sma[0, 1] - 20:roi5_sma[0, 1] + 10, :],
                            0.2)

                        a2, b2 = get_roi(
                            flow5[roi5_sma[1, 0] - 20:roi5_sma[1, 0] + 5, roi5_sma[1, 1] - 10:roi5_sma[1, 1] + 20, :],
                            0.2)

                        flow5_total1.append([a1 - x1, b1 - y1])
                        flow5_total2.append([a2 - x1, b2 - y1])
                        round1=5

                        gray_roi6 = gray[roi6_low:roi6_high , roi6_left:roi6_right ]
                        gray_roi7 = gray[roi7_low :roi7_high , roi7_left :roi7_right ]
                        flow6 = cv2.calcOpticalFlowFarneback(prevgray_roi6, gray_roi6, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                        flow7 = cv2.calcOpticalFlowFarneback(prevgray_roi7, gray_roi7, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                        a1, b1 = get_roi(flow6[round1:-round1, round1:-round1, :], 0.3)
                        a2, b2 = get_roi(flow7[round1:-round1, round1:-round1, :], 0.3)
                        flow6_total.append([a1 - x1, b1 - y1])
                        flow7_total.append([a2 - x1, b2 - y1])

                        # prevgray = gray
                        # prevgray_roi1 = gray_roi1
                        # prevgray_roi2 = gray_roi2
                        # prevgray_roi3 = gray_roi3
            if (k  == end):
                hh=end-start+1
                flow=np.copy(np.array(flow1_total))
                flow=np.vstack((flow,np.array(flow1_total1)))
                flow=np.vstack((flow,np.array(flow1_total2)))
                flow=np.vstack((flow,np.array(flow1_total3)))
                flow=np.vstack((flow,np.array(flow3_total)))
                flow=np.vstack((flow,np.array(flow3_total1)))
                flow=np.vstack((flow,np.array(flow3_total2)))
                flow=np.vstack((flow,np.array(flow3_total3)))
                flow=np.vstack((flow,np.array(flow4_total)))
                flow=np.vstack((flow,np.array(flow4_total1)))
                flow=np.vstack((flow,np.array(flow4_total2)))
                flow=np.vstack((flow,np.array(flow4_total3)))
                flow=np.vstack((flow,np.array(flow4_total4)))
                flow=np.vstack((flow,np.array(flow4_total5)))
                flow=np.vstack((flow,np.array(flow5_total1)))
                flow=np.vstack((flow,np.array(flow5_total2)))
                flow=np.vstack((flow,np.array(flow6_total)))
                flow=np.vstack((flow,np.array(flow7_total)))
                # print( pathp+"/"+str(k)+ ".npy")
                #np.save( pathp+"/"+str(k)+ ".npy", flow)

                totalflow=[]
                totalflowmic=[]
                totalflowmac = []
                # print(k)
                a=1
                totalflow,totalflowmic,totalflowmac=proce2(flow1_total,1.4,1.65,"left_eye",0,k,a,totalflow,totalflowmic,totalflowmac)
                totalflow,totalflowmic,totalflowmac=proce2(flow1_total1,1.4,1.65,"left_eye",1,k,a,totalflow,totalflowmic,totalflowmac)
                totalflow,totalflowmic,totalflowmac=proce2(flow1_total2,1.4,1.65,"left_eye",2,k,a,totalflow,totalflowmic,totalflowmac)
                totalflow,totalflowmic,totalflowmac=proce2(flow1_total3,1.4,1.65,"left_eye",3,k,a,totalflow,totalflowmic,totalflowmac)

                totalflow,totalflowmic,totalflowmac=proce2(flow3_total,1.4,1.65,"right_eye",0,k,a,totalflow,totalflowmic,totalflowmac)
                totalflow,totalflowmic,totalflowmac=proce2(flow3_total1,1.4,1.65,"right_eye",1,k,a,totalflow,totalflowmic,totalflowmac)
                totalflow,totalflowmic,totalflowmac=proce2(flow3_total2,1.4,1.65,"right_eye",2,k,a,totalflow,totalflowmic,totalflowmac)
                totalflow,totalflowmic,totalflowmac=proce2(flow3_total3,1.4,1.65,"right_eye",3,k,a,totalflow,totalflowmic,totalflowmac)

                totalflow,totalflowmic,totalflowmac=proce2(flow4_total,1.4,1.5,"mouth",0,k,a,totalflow,totalflowmic,totalflowmac)
                totalflow,totalflowmic,totalflowmac=proce2(flow4_total1,1.4,1.5,"mouth",1,k,a,totalflow,totalflowmic,totalflowmac)
                totalflow,totalflowmic,totalflowmac=proce2(flow4_total2,1.4,1.5,"mouth",2,k,a,totalflow,totalflowmic,totalflowmac)
                totalflow,totalflowmic,totalflowmac=proce2(flow4_total3,1.4,1.5,"mouth",3,k,a,totalflow,totalflowmic,totalflowmac)
                totalflow,totalflowmic,totalflowmac=proce2(flow4_total4,1.4,1.5,"mouth",4,k,a,totalflow,totalflowmic,totalflowmac)
                totalflow,totalflowmic,totalflowmac=proce2(flow4_total5,1.4,1.5,"mouth",5,k,a,totalflow,totalflowmic,totalflowmac)

                totalflow, totalflowmic, totalflowmac = proce2(flow5_total1, 1.4, 1.5, "nose", 1, k, a, totalflow,totalflowmic, totalflowmac)
                totalflow, totalflowmic, totalflowmac = proce2(flow5_total2, 1.4, 1.5, "nose", 2, k, a, totalflow,totalflowmic, totalflowmac)
                #
                # totalflow, totalflowmic, totalflowmac = proce2(flow6_total, 1.4, 1.7, "nose", 2, k, a, totalflow,totalflowmic, totalflowmac)
                # totalflow, totalflowmic, totalflowmac = proce2(flow7_total, 1.4, 1.7, "nose", 2, k, a, totalflow,totalflowmic, totalflowmac)
                # print(totalflow)

                #！！！！！！！！！只用一次nms试试 不好
                totalflow=np.array(nms2(totalflow))  #把所有通道融合起来
                totalflow=np.array(nms2(totalflow))
                totalflowmic_1=np.array(nms2(totalflowmic))
                totalflowmac_1=np.array(nms2(totalflowmac))

                # print(str(k - hh) + "--" + str(k) + "all:")
                # print(totalflow)
                totalflow_1=totalflow-(k - hh)
                move=100
                for i in range(len(totalflow_1)):
                    # if (totalflow_1[i, 0] - (k - hh) < 175):
                    if (totalflow_1[i, 0] < 100  and totalflow_1[i, 1] > 100 ):
                        if(totalflow_1[i, 1]<150):
                            move=totalflow_1[i, 1]+20
                        elif(totalflow_1[i, 0]>50):
                            move=totalflow_1[i, 0]-20
                        else:
                            a=min(189,totalflow_1[i, 1])
                            #a = min(155, totalflow_1[i, 1])
                            move=a+10


                # if(len(totalflowmic_1)>0):
                #     print("micro:")
                #     for i in range(len(totalflowmic_1)):
                #         print(totalflowmic_1 + (k - hh))
                # if (len(totalflowmac_1) > 0):
                #     print("macro:")
                #     for i in range(len(totalflowmac_1)):
                #         print(totalflowmac_1 + (k - hh))
                # print(totalflow.shape)
                # print(lable_vio.shape)
                # print(k-hh)
                # print(k)

                # for m in range(hh):
                #     if(totalflow[m]==1):
                #         lable_vio[k-hh+m]=1
                lable_vio=np.vstack((lable_vio,totalflow))
                # lable_vio=np.vstack((lable_vio,totalflowmac_1))
                break
    print("全部：")

    lable_video_update=[] #去除一些太短的片段
    lable_video_update1 = []
    for i in range(len(lable_vio)):
        if(lable_vio[i,1]-lable_vio[i,0]>=10):
            lable_video_update.append([lable_vio[i,0],lable_vio[i,1]])
    lable_video_update=np.array(nms2(lable_video_update))

    lable_video_update = np.array(nms2(lable_video_update))
    for i in range(len(lable_video_update)):
        if(lable_video_update[i,1]!=0):
            lable_video_update1.append([lable_video_update[i,0],lable_video_update[i,1]])
    lable_video_update1=np.array(lable_video_update1)
    # print(lable_video_update1)
    return lable_video_update1