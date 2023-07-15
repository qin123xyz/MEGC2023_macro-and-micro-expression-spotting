
import os
import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
#import spot_util_samm as fl
# import spot_util_samm as SAMM
import spot_util_samm_val as SAMM
import xlrd
import csv

from xlrd import xldate_as_tuple
from time import *
import pickle
import multiprocessing

path_SAMM="/space0/qinwf/MEdata/SAMM long videos/SAMM_longvideos/"
    #"D:/graduate/medata/SAMM long videos/SAMM_longvideos/"
    #"C:/sheng/SAMM/SAMM_longvideos/"
path_label="/space0/qinwf/MEdata/SAMM long videos/SAMM_LongVideos_V3_Release.xlsx"
    #"D:/graduate/medata/SAMM long videos/SAMM_LongVideos_V3_Release.xlsx"
    #'C:/sheng/SAMM/SAMM_LongVideos_V3_Release.xls'
def tes_SAMM(path5,flow):
    print("path5即vio")
    print(path5)
    data1 = xlrd.open_workbook(path_label)
    table = data1.sheets()[0]
    # 创建一个空列表，存储Excel的数据
    lable_vio = []
    num_micro = 0
    for j in range(len(flow)):
        start2 = flow[j, 0] + 1
        end2 = flow[j, 1] + 1
        if (end2 - start2 <= 100):# <=100 算是micro  200fps   0.5s
            print("lll")
            num_micro += 1
    for rown in range(table.nrows):  # nrows代表行
        vio = table.cell_value(rown, 1)  #每一行的第2列  006_1_1

        if(vio[0:5]==path5):#前5个字符  006_1
            start = int(table.cell_value(rown, 3) )
            mid=int(table.cell_value(rown, 4))
            end=int(table.cell_value(rown, 5))
            if(end==0):
                end=mid+20
            lable_vio.append([start,end])


    lable_vio=np.array(lable_vio)
    true_lable=0
    true_lable_03=0
    true_lable_02=0
    true_lable_04=0
    true_lable_mic=0
    true_lable_mic_03 = 0
    true_lable_mic_02 = 0
    true_lable_mic_04 = 0
    FPPP = []
    #？？？？？？关键是没有判断end2-start2是否小于100  当label end1-start1<=100时，并满足IOU,直接认定是TPmicro了
    #其实也可以  如果预测的段不小于100  很大的话 IOU也很难满足
    for i in range(len(lable_vio)):##########gt在外层循环  对每个gt找满足IOU的预测的段
        start1 = lable_vio[i, 0]   #是间断的标签
        end1 = lable_vio[i, 1]
        percent=0

        for j in range(len(flow)):
            start2=flow[j,0]+1   #预测的标签
            end2 = flow[j,1]+1
            if(not(end2<start1 or start2>end1)):#end2<start1 or start2>end1没交集
                min_start=min(start1,start2)
                max_start=max(start1,start2)
                min_end = min(end1, end2)
                max_end = max(end1, end2)
                percent=(float(min_end-max_start))/(max_end-min_start)
                print(percent)


                if(percent>=0.5):
                    FPPP.append(j)
                    print("lable:{},{},{}  test:{},{},{} percent={}".format(start1,end1,end1-start1,start2,end2,end2-start2,percent))
                    true_lable += 1 #统计所有正确的表情段数  到时候减去micro 就是macro
                    true_lable_04 += 1
                    true_lable_03 += 1
                    true_lable_02 += 1
                    if(end1-start1<=100):#label也是按长度区分微表情和宏表情   小于100 micro
                        true_lable_mic += 1
                        true_lable_mic_03+=1
                        true_lable_mic_02 += 1
                        true_lable_mic_04 += 1
                        print("正确的微表情")
                    with open("./valspotsammre/val_spotutilsamm.csv", "a", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([ path5, start1, end1, start2, end2, "TP"])


                    break####!!!!!!!!!!!!!!!!!!!!!!注意  对一个gt而言  找到一个满足IOU的段就ok  break
                if (percent >= 0.4):
                    true_lable_04 += 1
                    true_lable_03 += 1
                    true_lable_02 += 1
                    if (end1 - start1 <= 100):
                        true_lable_mic_03 += 1
                        true_lable_mic_02 += 1
                        true_lable_mic_04+= 1
                    break
                if (percent >= 0.3):
                    true_lable_03 += 1
                    true_lable_02 += 1
                    if (end1 - start1 <= 100):

                        true_lable_mic_03 += 1
                        true_lable_mic_02 += 1

                    break
                if (percent >= 0.2):
                    true_lable_02 += 1
                    if (end1 - start1 <= 100):

                        true_lable_mic_02 += 1

                    break


        if(percent<0.5):
            print("lable:{},{} 没有正确结果".format(start1,end1))
            with open("./valspotsammre/val_spotutilsamm.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([path5, start1, end1, '', '', "FN"])
    FPPP=set(FPPP)
    FPPP = list(FPPP)
    for j in range(len(flow)):
        if j not in FPPP:
            start2 = flow[j, 0] + 1  # 预测的标签
            end2 = flow[j, 1] + 1
            with open("./valspotsammre/val_spotutilsamm.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([path5, '', '', start2, end2, "FP"])

    print("正确的数量：{}".format(true_lable))
    print("测试出的数量：{}".format(len(flow)))
    print("全部的微表情数量：{}".format(len(lable_vio)))
    return num_micro,true_lable,true_lable_02,true_lable_03,true_lable_04,true_lable_mic,true_lable_mic_04,true_lable_mic_03,len(flow),len(lable_vio)


def worker(vio, i_worker):
    allvio_rightlable=0
    allvio_rightlable_02=0
    allvio_rightlable_03=0
    allvio_rightlable_04=0
    allvio_rightlable_mic=0
    allvio_rightlable_mic_02=0
    allvio_rightlable_mic_03=0
    all_num_micro=0
    allvio_test=0
    alllable_num=0

    path5 = vio
    # SAMM
    pp = SAMM.draw_roiline19(path_SAMM, path5, 6, -4, 7)  # 18是直接使用光流计算，19是全部，20是去掉全局移动
    pp = pp * 7#所有的段的帧序号都*7 比如start是2 实际对应是第14帧
    print("this vio 预测段:{}".format(pp))

    num_micro, true_lable, true_lable_02, true_lable_03, true_lable_04, true_lable_mic, true_lable_mic_02, true_lable_mic_03, test_true, lable_num = tes_SAMM(
        path5, pp)

    allvio_rightlable += true_lable
    allvio_rightlable_02 += true_lable_02
    allvio_rightlable_03 += true_lable_03
    allvio_rightlable_04 += true_lable_04
    allvio_rightlable_mic += true_lable_mic
    allvio_rightlable_mic_03 += true_lable_mic_03
    allvio_rightlable_mic_02 += true_lable_mic_02
    allvio_test += test_true
    alllable_num += lable_num
    all_num_micro += num_micro
    print("在" + vio + "视频中有{}个正确的分析".format(true_lable))

    result = [allvio_rightlable,
              allvio_rightlable_02,
              allvio_rightlable_03,
              allvio_rightlable_04,
              allvio_rightlable_mic,
              allvio_rightlable_mic_02,
              allvio_rightlable_mic_03,
              all_num_micro,
              allvio_test,
              alllable_num,
              ]

    with open(f'./valspotsammre/samm_rst{i_worker}.pkl', 'wb') as f:
        pickle.dump(result, f)


def multivideo_SAMM(path4):
    fileList = os.listdir(path4) #path4:D:\graduate\medata\SAMM long videos\SAMM_longvideos
                                 #filelist就是006_1  006_2.....
    k=0
    allvio_rightlable=0
    allvio_rightlable_02=0
    allvio_rightlable_03=0
    allvio_rightlable_04=0
    allvio_rightlable_mic=0
    allvio_rightlable_mic_02=0
    allvio_rightlable_mic_03=0
    all_num_micro=0
    allvio_test=0
    alllable_num=0
    print("lllllllllllll")

    n_result = len(fileList)
    # n_worker = n_result
    n_worker = 8

    pool = multiprocessing.Pool(n_worker)
    for i_worker, vio in enumerate(fileList):
        worker(vio, i_worker)
        # if(True):
        # if(vio=="020_2"):
        # if(vio=="009_3"):
        pool.apply_async(worker, (vio, i_worker))
    pool.close()
    pool.join()

    gather_result = []
    for i_worker in range(n_result):
        with open(f'./valspotsammre/samm_rst{i_worker}.pkl', 'rb') as f:
            gather_result.append(pickle.load(f))
    gather_result = np.array(gather_result).sum(axis=0)

    allvio_rightlable=gather_result[0]
    allvio_rightlable_02=gather_result[1]
    allvio_rightlable_03=gather_result[2]
    allvio_rightlable_04=gather_result[3]
    allvio_rightlable_mic=gather_result[4]
    allvio_rightlable_mic_02=gather_result[5]
    allvio_rightlable_mic_03=gather_result[6]
    all_num_micro=gather_result[7]
    allvio_test=gather_result[8]
    alllable_num =gather_result[9]

    print("-------------------------")
    print("-------------------------")
    print("-------------------------")
    print("共有{}个正确的分析".format(allvio_rightlable))
    print("共有{}个大于百分之20的分析".format(allvio_rightlable_02))
    print("共有{}个大于百分之30的分析".format(allvio_rightlable_03))
    print("共有{}个大于百分之40的分析".format(allvio_rightlable_04))
    print("共有{}个正确的微表情的分析".format(allvio_rightlable_mic))
    print("共有{}个大于百分之30的正确的微表情的分析".format(allvio_rightlable_mic_03))
    print("共有{}个大于百分之20的正确的微表情的分析".format(allvio_rightlable_mic_02))
    print("共有{}个正确的分析".format(allvio_rightlable))
    print("共测试出{}个分析".format((allvio_test)))
    print("共有{}个微表情".format(alllable_num))
    print("共测试出有{}个微表情".format(all_num_micro))
    print("共测试出{}宏表情".format(allvio_test - all_num_micro))

    print("------------------")
    print("------------------")
    print("------------------")
    print("iou=0.5")
    P = (allvio_rightlable) / (allvio_test)  # 准确率
    R = (allvio_rightlable) / (alllable_num)  # 召回率
    F = (2 * P * R) / (P + R)
    print("计算P系数,准确率：" + str(P))
    print("计算R系数,召回率：" + str(R))
    print("计算F系数,综合评价：" + str(F))

    print("iou=0.5 for mic")
    P = (allvio_rightlable_mic) / (all_num_micro)  # 准确率
    R = (allvio_rightlable_mic) / (159)  # 召回率
    F = (2 * P * R) / (P + R)
    print("计算P系数,准确率：" + str(P))
    print("计算R系数,召回率：" + str(R))
    print("计算F系数,综合评价：" + str(F))

    print("iou=0.5 for mac")
    P = (allvio_rightlable - allvio_rightlable_mic) / (allvio_test - all_num_micro)  # 准确率
    R = (allvio_rightlable - allvio_rightlable_mic) / (343)  # 召回率
    F = (2 * P * R) / (P + R)
    print("计算P系数,准确率：" + str(P))
    print("计算R系数,召回率：" + str(R))
    print("计算F系数,综合评价：" + str(F))

    print("------------------")
    print("------------------")
    print("------------------")
    print("iou=0.4")
    P = (allvio_rightlable_04) / (allvio_test)  # 准确率
    R = (allvio_rightlable_04) / (alllable_num)  # 召回率
    F = (2 * P * R) / (P + R)
    print("计算P系数,准确率：" + str(P))
    print("计算R系数,召回率：" + str(R))
    print("计算F系数,综合评价：" + str(F))

    print("iou=0.4 for mic")
    P = (allvio_rightlable_mic_02) / (all_num_micro)  # 准确率
    R = (allvio_rightlable_mic_02) / (159)  # 召回率
    F = (2 * P * R) / (P + R)
    print("计算P系数,准确率：" + str(P))
    print("计算R系数,召回率：" + str(R))
    print("计算F系数,综合评价：" + str(F))

    print("iou=0.4 for mac")
    P = (allvio_rightlable_04 - allvio_rightlable_mic_02) / (allvio_test - all_num_micro)  # 准确率
    R = (allvio_rightlable_04 - allvio_rightlable_mic_02) / (343)  # 召回率
    F = (2 * P * R) / (P + R)
    print("计算P系数,准确率：" + str(P))
    print("计算R系数,召回率：" + str(R))
    print("计算F系数,综合评价：" + str(F))

    print("------------------")
    print("------------------")
    print("------------------")
    print("iou=0.3")
    P = (allvio_rightlable_03) / (allvio_test)  # 准确率
    R = (allvio_rightlable_03) / (alllable_num)  # 召回率
    F = (2 * P * R) / (P + R)
    print("计算P系数,准确率：" + str(P))
    print("计算R系数,召回率：" + str(R))
    print("计算F系数,综合评价：" + str(F))

    print("iou=0.3 for mic")
    P = (allvio_rightlable_mic_03) / (all_num_micro)  # 准确率
    R = (allvio_rightlable_mic_03) / (159)  # 召回率
    F = (2 * P * R) / (P + R)
    print("计算P系数,准确率：" + str(P))
    print("计算R系数,召回率：" + str(R))
    print("计算F系数,综合评价：" + str(F))

    print("iou=0.3 for mac")
    P = (allvio_rightlable_03 - allvio_rightlable_mic_03) / (allvio_test - all_num_micro)  # 准确率
    R = (allvio_rightlable_03 - allvio_rightlable_mic_03) / (343)  # 召回率
    F = (2 * P * R) / (P + R)
    print("计算P系数,准确率：" + str(P))
    print("计算R系数,召回率：" + str(R))
    print("计算F系数,综合评价：" + str(F))
    P=(allvio_rightlable)/(allvio_test)#准确率
    R=(allvio_rightlable)/(alllable_num)#召回率
    F=(2*P*R)/(P+R)
    print("计算P系数,准确率："+str(P))
    print("计算R系数,召回率："+str(R))
    print("计算F系数,综合评价："+str(F))

if __name__ == '__main__':
    multivideo_SAMM(path_SAMM)