import os

import numpy as np
import xlrd

import samm_util as SAMM
import cas_util_chu as CAS
from time import *
import pandas as pd
t_iou =0.5

def detect_expression(path,name,fps,path_label):

    global pp
    mic_default_d = int(fps * 2 / 3)
    data_xls = xlrd.open_workbook(path_label)
    table_xls = data_xls.sheets()[0]
    labels = {}

    mic_default_d = int(fps * 2 / 3)
    for i_row in table_xls.get_rows():
        key = str(i_row[0].value) + '_' + str(i_row[1].value)  # spNO.1_a
        if key not in labels.keys():
            labels[key] = []
        # print("type:{}".format(type(i_row[3].value)))
        labels[key].append([i_row[2].value,  i_row[4].value])

    #labels把subnum和段开始结束对应上  之后根据vionum即key 找到对应gt段
    subs = os.listdir(path)
    n_gt=0
    n_gt_mic=0
    n_pred=0
    n_pred_mic=0
    tp_all = 0
    tp_mic = 0
    mic_frame =5 # fps // 2
    pathsave = "./valcas3_subcsv"
    if not os.path.exists(pathsave):
        os.mkdir(pathsave)
    for sub in subs: #spno1  spno2
        if sub[-4:]=='.zip':
            continue
        # if sub!='spNO.161':
        #     continue
        if sub=='spNO.165':
            continue
        if sub == 'spNO.147' or sub=='spNO.194' or sub=='spNO.154' or sub=='spNO.8' or sub=="spNO.162" or sub=='spNO.210'\
                or sub=='spNO.202' or sub=='spNO.184' or sub=='spNO.198' or sub=='spNO.155'  or sub=='spNO.190'or sub=='spNO.166'\
                or sub=='spNO.201' or sub=='spNO.7'or sub=='spNO.171'or sub=='spNO.149'or sub=='spNO.189'or sub=='spNO.173'or sub=='spNO.159'\
                or sub=='spNO.139' or sub=='spNO.216'or sub=='spNO.145':
            continue
        if sub=='spNO.152':  #直接无landmark
            continue
        if os.path.exists(pathsave + "/" + str(sub) + ".csv"):
            continue
        print("sub:{}".format(sub))
        subpath = os.path.join(path, sub)
        df= pd.DataFrame(columns=["sub", "tp", "pre", "gt","precision","recall","f1",
                                  "tp_mic","pre_mic","gt_mic","precision_mic","recall_mic","f1_mic",
                                  "tp_mac","pre_mac","gt_mac","precision_mac","recall_mac","f1_mac"])
        for vio in os.listdir(subpath):  #a b d e
            subnum=sub+'_'+vio
            # if subnum != 'spNO.161_f':
            #     continue
            # if subnum!='spNO.165_l':
            #     continue
            # if subnum=='spNO.165_m':
            #     continue
            if subnum=='spNO.203_e':
                continue
            if subnum not in labels.keys():
                print("excel without label:{}".format(subnum))
                continue
            viopath=os.path.join(subpath,vio)
            colorpath=os.path.join(viopath,'color')
            print("colorpath:{}".format(colorpath))
            violabel = labels[subnum]
            if name == "samm":
                pp = SAMM.draw_roiline19(viopath,'color', 6, -4, 7)
                pp = pp * 7
            elif name == "cas":
                pp = CAS.draw_roiline19(viopath,'color', 0, -4, 1)
            pp = pp.tolist()  # 存的是所有的段
            n_pred+=len(pp)
            for ps,pe in pp:
                if pe - ps <= mic_frame:
                    n_pred_mic += 1
            pp.sort()
            print("this vio gt段:{}".format(violabel))  # [1195.0,1266.0]
            print("this vio 预测段:{}".format(pp))  # 这个vio预测的输出全部段

            #根据label和预测的这些段  计算IOU  一个gt对应一个TP???  or 多个
            for label_start, label_end in violabel:
                n_gt+=1
                label_start = int(label_start)
                label_end = int(label_end)
                if label_end-label_start<=mic_frame:
                    n_gt_mic+=1
                for j, (predict_start, predict_end) in enumerate(pp):
                    if not (predict_end < label_start or predict_start > label_end):
                        all_points = sorted([label_start, label_end, predict_start, predict_end])
                        percent = (float(all_points[2] - all_points[1])) / (all_points[3] - all_points[0])
                        if percent >= 0.5:
                            tp_all+= 1
                            print("TP")
                            # ！！！！！注意这里评价为tp的标准是 label段长度小于micframe就是label micro
                            if label_end - label_start <= mic_frame:  # ！！！！！！！！！！label的micro不是读的excel 而是按照长度mic_frame 区分
                                tp_mic+= 1



        precision = tp_all / n_pred
        recall = tp_all / n_gt
        f1 = (2 * precision * recall) / (precision + recall)
        precision_mic = 0
        if n_pred_mic != 0:
            precision_mic = tp_mic / n_pred_mic
        recall_mic=0
        if n_gt_mic!=0:
            recall_mic=tp_mic/n_gt_mic
        f1_mic=0
        if precision_mic+recall_mic!=0:
            f1_mic=(2*precision_mic*recall_mic)/(precision_mic+recall_mic)

        tp_mac=tp_all-tp_mic
        n_gt_mac = n_gt - n_gt_mic
        n_pred_mac=n_pred-n_pred_mic
        precision_mac = 0
        if n_pred_mac != 0:
            precision_mac = tp_mac / n_pred_mac
        recall_mac = tp_mac / n_gt_mac

        f1_mac = (2 * precision_mac * recall_mac) / (precision_mac + recall_mac)
        df.loc[0,"sub"] = sub
        df.loc[0,"tp"] = tp_all
        df["pre"] = n_pred
        df["gt"] = n_gt
        df["precision"] =precision
        df["recall"]=recall
        df["f1"]=f1

        df["tp_mic"] = tp_mic
        df["pre_mic"] = n_pred_mic
        df["gt_mic"] = n_gt_mic
        df["precision_mic"] = precision_mic
        df["recall_mic"] = recall_mic
        df["f1_mic"] = f1_mic

        df["tp_mac"] = tp_mac
        df["pre_mac"] = n_pred_mac
        df["gt_mac"] = n_gt_mac
        df["precision_mac"] = precision_mac
        df["recall_mac"] = recall_mac
        df["f1_mac"] = f1_mac

        df.to_csv(pathsave+"/"+str(sub)+".csv",index=False)

def main():

    #path_samm ="D:\\graduate\\medata\\SAMM_Test_cropped\\"
   # detect_expression(path_samm,'samm',200)
    path_cas ='/space/qwf/casme^3/part_A/data/Compressed_version1_seperate_compress'
        #"D:/graduate/medata/casmesqr/rawpic"
        #"/space0/qinwf/DATA/casmesqr/rawpic"
        #"D:/graduate/medata/casmesqr/rawpic"
        #"D:\\graduate\\medata\\CAS_Test_cropped\\"

    label_cas='/space0/qinwf/MEdata/cas^3/CAS(ME)3_part_A_v1.xls'
        #"D:/graduate/medata/casmesqr/CAS(ME)^2code_final(Updated).xlsx"
        #"/space0/qinwf/DATA/casmesqr/CAS(ME)^2code_final(Updated).xlsx"
        #"D:/graduate/medata/casmesqr/CAS(ME)^2code_final(Updated).xlsx"
    detect_expression(path_cas,'cas',30,label_cas)


if __name__ == '__main__':
    main()
