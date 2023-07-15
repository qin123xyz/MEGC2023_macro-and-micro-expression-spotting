import csv
import multiprocessing
import os
import pickle
import shutil

import numpy as np
import xlrd
import cas_util as CAS

def read_casme2_video(video_path):
    start_sub, end_sub = 3, -4  #前2帧  后三帧都不要
    files = [i for i in os.listdir(video_path) if 'jpg' in i]
    files.sort(key=lambda x: int(x[start_sub:end_sub]))
    files = [os.path.join(video_path, i) for i in files]
    return files

def read_casme2_label(label_path, fps):
    # 宏观表情平均持续39帧（30fps）
    mic_default_d = int(fps * 2 / 3)

    data_xls = xlrd.open_workbook(label_path)
    table_xls = data_xls.sheets()[0]

    labels = {}
    s = [15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40]
    dir = {"disgust1": "0101", "disgust2": "0102", "anger1": "0401", "anger2": "0402", "happy1": "0502",
           "happy2": "0503", "happy3": "0505", "happy4": "0507", "happy5": "0508"}
    # for i_row in table_xls:
    #     key = i_row[10].value
    for i_row in table_xls.get_rows():
        snum = s[int(i_row[0].value) - 1]
        cla = i_row[1].value.split("_")[0]
        clanum = dir[cla]
        subnum = str(snum) + '_' + clanum  # 15_0401
        # print(subnum)
        key = subnum
        if key not in labels:
            labels[key] = []
        labels[key].append([int(i_cell.value) for i_cell in i_row[2:5]])

    labels = {k:  np.array([[i_row[0], i_row[2] if i_row[2] else i_row[1] + mic_default_d] for i_row in v])
              for k, v in labels.items()}
    return labels
def metrics(label_slice, predict_slice, mic_frame, dataset_root_path, debug_message=True):
    n_pre_micro = int(((predict_slice[:, 1] - predict_slice[:, 0]) <= mic_frame).sum()) if predict_slice.shape[0] else 0
   #按照长度mic_frame 区分微表情
    tp = 0
    tp_mic = 0

    # 修正标注1开始，预测0开始
    predict_slice += 1

    FPPP = []
    for label_start, label_end in label_slice.tolist():
        percent = 0
        for j, (predict_start, predict_end) in enumerate(predict_slice.tolist()):
            if not (predict_end < label_start or predict_start > label_end):

                all_points = sorted([label_start, label_end, predict_start, predict_end])
                percent = (float(all_points[2] - all_points[1])) / (all_points[3] - all_points[0])





                if percent >= 0.5:
                    tp += 1
                    # ！！！！！注意这里评价为tp的标准是 label段长度小于micframe就是label micro
                    if label_end - label_start <= mic_frame:  # ！！！！！！！！！！label的micro不是读的excel 而是按照长度mic_frame 区分
                        tp_mic+= 1
                    FPPP.append(j)
                    if debug_message:
                        print(f'lable:{label_start},{label_end}  test:{predict_start},{predict_end} percent={percent}')

                    with open('./valcas^2re/valcas_sqr.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([dataset_root_path, label_start, label_end, predict_start, predict_end, 'TP'])
                    break

        if percent < 0.5:  #一个gt与所有的预测段 算IOU 都没有大于0.5的
            if debug_message:
                print(f'lable:{label_start},{label_end} 没有正确结果')
            with open('./valcas^2re/valcas_sqr.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([dataset_root_path, label_start, label_end, '', '', 'FN'])

    FPPP = list(set(FPPP))
    for j, (predict_start, predict_end) in enumerate(predict_slice):
        if j not in FPPP:
            with open('./valcas^2re/valcas_sqr.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([dataset_root_path, '', '', predict_start, predict_end, 'FP'])

    return tp, tp_mic, len(label_slice), len(predict_slice), n_pre_micro

def casme2_worker( sub,  i_worker, debug_message=False):
    # dataset_root_path = "D:/graduate/medata/casmesqr/rawpic"
    # # 'C:\\sheng\\casme2\\rawpic'
    # label_path = "D:/graduate/medata/casmesqr/CAS(ME)^2code_final(Updated).xlsx"
    dataset_root_path = '/space0/qinwf/MEdata/casmesqr/rawpic'#'/space0/qinwf/MEdata/openface_cas^2'#
        #"/space0/qinwf/DATA/casmesqr/rawpic"
    # "D:/graduate/medata/casmesqr/rawpic"
    label_path = '/space0/qinwf/MEdata/casmesqr/CAS(ME)^2code_final(Updated).xlsx'
        #"/space0/qinwf/DATA/casmesqr/CAS(ME)^2code_final(Updated).xlsx"
    # 'CAS(ME)^2code_final.xls'

    fps = 30
    mic_frame = fps // 2   #15帧  低于15帧的label  按照micro  0.5s  EXCEL里<=15的都是mic
    result = []

    sub_path = os.path.join(dataset_root_path, sub)
    videos = os.listdir(sub_path)
    labels = read_casme2_label(label_path, fps)
    for i_video in videos:
        # CAS(ME)^2
        #files = read_casme2_video(os.path.join(dataset_root_path, sub, i_video))#s15\15_0101disgustingteeth
        predict = CAS.draw_roiline19(sub_path , i_video , 3, -4,1)
        pp=predict.tolist()
        vionum = i_video[0:7]  # 15_0101
        print("vionum:{}".format(vionum))
        print("this vio 预测段:{}".format(pp))  # 这个vio预测的输出全部段
        if vionum not in labels.keys():  # !!!!!!!!!!!!!!!!!!!!!!  115_0508这个excel没有label 没有1 happy5
            print("excel without label:{}".format(vionum))
            continue
        gt = labels[vionum]
        print("this vio gt段:{}".format(gt))

        i_video_relative = sub + '_' + i_video
        # gt = labels[i_video_relative] if i_video_relative in labels else np.array([])
        result.append(metrics(gt, predict, mic_frame, i_video_relative))

    result = [np.array([j[i] for j in result]).sum(axis=0) for i in range(5)]

    with open(f'./valcas^2re/casmesqr_{i_worker}.pkl', 'wb') as f:
        pickle.dump(result, f)
        #pickle可以将对象数据压到一个文件中，永久保存。这样在取用时，只需将该文件中的数据取出。而不是每次都重新进过各种语句，处理得到对象数据。

    if debug_message:
        print(f'{i_worker} finished')
def report( n_gt, n_pred, tp, prefix='', show_message=True):
    precision = tp / n_pred
    recall = tp / n_gt
    f1 = (2 * precision * recall) / (precision + recall)

    if show_message:
        print('------------------')
        print(f"{prefix}精准率：：{precision}")
        print(f"{prefix}召回率:{recall}")
        print(f"{prefix}F1系数:{f1}")
        # print(f"{prefix}精准率：{','.join([f'{i:20}' for i in precision])}")
        # print(f"{prefix}召回率：{','.join([f'{i:20}' for i in recall])}")
        # print(f"{prefix}F1系数：{','.join([f'{i:20}' for i in f1])}")
    return precision, recall, f1
def main_casme2( show_message=True):
    dataset_root_path = "/space0/qinwf/MEdata/casmesqr/rawpic"
        #"/space0/qinwf/DATA/casmesqr/rawpic"
        #"D:/graduate/medata/casmesqr/rawpic"
    sub_list = os.listdir(dataset_root_path)#s15 s16 s19...
    # sub_list =['s15','s16']
    n_result = len(sub_list)#22
    n_worker = min(25, n_result)
    #
    pool = multiprocessing.Pool(n_worker)#22个sub 22个进程
    for i_worker, i_sub in enumerate(sub_list):
        casme2_worker(i_sub,  i_worker)
        pool.apply_async(casme2_worker, (i_sub,  i_worker))
            #apply_async是异步非阻塞的 不用等待当前进程执行完毕，随时根据系统调度来进行进程切换
    pool.close()
    pool.join()

    gather_result = []
    for i_worker in range(n_result):
        with open(f'./valcas^2re/casmesqr_{i_worker}.pkl', 'rb') as f:
            gather_result.append(pickle.load(f))

    tp = np.array([i[0] for i in gather_result]).sum(axis=0)
    tp_mic = np.array([i[1] for i in gather_result]).sum(axis=0)
    tp_mac = tp-tp_mic
    n_gt = np.array([i[2] for i in gather_result]).sum(axis=0)
    n_pred = np.array([i[3] for i in gather_result]).sum(axis=0)
    n_pred_mic = np.array([i[4] for i in gather_result]).sum(axis=0)
    n_pred_mac = n_pred - n_pred_mic

    n_gt_mic = 57

    n_gt_mac = 300
    if show_message:
        print('-------------------------')

        print(f'共有{tp}个正确的分析，共有{tp_mic}个正确的微表情分析')
        print('-------------------------')
        print(f'共有表情{n_gt}个')
        print(f'共测试出{n_pred}个')
        print(f'宏表情有{n_pred_mac}个')
        print(f'微表情有{n_pred_mic}个')

    report(n_gt, n_pred, tp, '', show_message=show_message)
    report( n_gt_mac, n_pred_mac, tp_mac, '宏表情',  show_message=show_message)
    report( n_gt_mic, n_pred_mic, tp_mic, '微表情',  show_message=show_message)
if __name__ == '__main__':
    main_casme2()