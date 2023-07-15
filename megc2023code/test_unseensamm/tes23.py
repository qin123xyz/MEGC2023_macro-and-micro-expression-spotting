import os

import spot_util_casme as CAS
import spot_util_samm as SAMM
#import spot_util_sammhuifu705yuan as SAMM
from time import *
import pandas as pd
def read_casme2_video(video_path):
    #start_sub, end_sub = 3, -4
    files = [i for i in os.listdir(video_path)]
    #files.sort(key=lambda x: int(x[start_sub:end_sub]))
    files.sort(key=lambda x: int(x.split('_')[3][0:-4]))
    files = [os.path.join(video_path, i) for i in files]
    return files
def detect_expression(path,name,fps):
    global pp
    fileList = os.listdir(path)
   # print(fileList)
    data = pd.DataFrame(columns=["vid","onset","offset","type"])
    j=0
    iou = 0.5
    for vio in fileList:
            if os.path.isfile(path + '/' + vio):
                continue
            elif os.path.isdir(path + '/' + vio):
                print(vio)
            if vio[-4:] == ".zip":
                continue
            if name == "samm":
                # if vio!="002_4":
                #     continue
                pp = SAMM.draw_roiline19(path , vio , 6, -4,7)
                #pp=SAMMTVL1.draw_roiline19(path , vio , 6, -4,7)

                pp=pp*7
            elif name == "cas":

                face_size = 256
                flow_scale = face_size / 256  # 1
                process_kwargs = {
                    'fps': fps,  # frame
                    'l_expand': fps,  # frame le
                    'l_small_expend': int(fps / 3),  # frame lse
                    'l_split': int(fps * 2 / 3),  # frame ls
                    'bound_ignore': int(fps * 4 / 15),  # frame
                    't_peak_valley_ratio': 0.33,  # ratio r1
                    't_peak_ratio': 0.33,  # ratio r2
                    't_valley_gap': 0.3 * flow_scale,  # pixel vg
                    't_peak_relative_inf': 1.4 * flow_scale,  # pixel pri
                    't_peak_inf': 0.7 * flow_scale,  # pixel pi
                    't_ext_gap': 0.8 * flow_scale,  # pixel emd signal gap
                    'frequency_inf': 1,  #
                    'frequency_sup': 5,  #
                }
                le_p_kwargs = process_kwargs.copy()
                le_p_kwargs['t_flow_gap'] = 1.8 * flow_scale
                re_p_kwargs = process_kwargs.copy()
                re_p_kwargs['t_flow_gap'] = 1.8 * flow_scale
                mth_p_kwargs = process_kwargs.copy()
                mth_p_kwargs['t_flow_gap'] = 1.85 * flow_scale
                ns_p_kwargs = process_kwargs.copy()
                ns_p_kwargs['t_flow_gap'] = 2.1 * flow_scale
                ll_p_kwargs = process_kwargs.copy()
                rl_p_kwargs = process_kwargs.copy()
                t_flow_percent = {
                    't_tip_p': 0.7,
                    't_le_p': 0.2,
                    't_re_p': 0.3,
                    't_mth_tp': 0.3,
                    't_mth_pp': 0.2,
                    't_ns_p': 0.2,
                    't_ll_p': 0.3,
                    't_rl_p': 0.3,
                }
                files = read_casme2_video(os.path.join(path, vio))  # s15\15_0101disgustingteeth
                #print(files)
                pp = CAS.draw_roiline19(files, le_p_kwargs, re_p_kwargs, mth_p_kwargs, ns_p_kwargs,
                                    ll_p_kwargs, rl_p_kwargs, t_flow_percent, face_size)

                #pp = CAS.draw_roiline19(path , vio , 0, -4,1)


            pp = pp.tolist()#存的是所有的段
            pp.sort()
            print(pp)#输出全部段
            i=0
            for i,interval in enumerate(pp):
                print("第i:{}个预测段".format(i))
                data.loc[i+j,'vid'] = vio
                data.loc[i+j,'onset'] = interval[0]
                data.loc[i+j,'offset'] = interval[1]
                if (interval[1]-interval[0])/fps > iou: #长度/30>0.5  长度大于15帧的是mae    /200>0.5   >100
                    data.loc[i+j,'type'] = 'mae'
                else:
                    data.loc[i+j,'type'] = 'me'
            print("i:".format(i))
            j=i+j+1
    data.to_csv(name+'_pred'+".csv",index=False)

def main():
    path_samm ='/space/qwf/sammtest/SAMM_data'
        #'E:/MEGC2023 test/SAMM_data'  #'/space/qwf/sammtest/SAMM_data/'
        #"D:/graduate/medata/SAMM_Test_cropped/"
    #/home/data2/CZP/2022MEGC-SPOT/MEGC2022_testSet/SAMM_Test_cropped/"
    detect_expression(path_samm,'samm',200)
    # path_cas ='E:/MEGC2023 test/openface_cas3align'
    # # "D:\\graduate\\medata\\CAS_Test_cropped\\"
    # # #     #"/home/data2/CZP/2022MEGC-SPOT/MEGC2022_testSet/CAS_Test_cropped/"
    # detect_expression(path_cas,'cas',30)


if __name__ == '__main__':
    main()
