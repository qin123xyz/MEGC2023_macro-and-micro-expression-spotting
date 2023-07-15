import os
import samm_util as SAMM
#import cas_util as CAS
# import cas_huifu702 as CAS
import  casyuan_huifu as CAS
#import cas_utill_TVL1 as CAS
from time import *
import pandas as pd
from drawroiwinflow import draw_roiwinflow
#注意test和val切换的时候  不要忘记util里    filestsort不一样！！！！！！！！！！
def detect_expression(path,name,fps):
    global pp, i
    fileList = os.listdir(path)
    print(fileList)
    data = pd.DataFrame(columns=["vid", "onset", "offset", "type"])
    #data = pd.DataFrame(columns=["vid","pred_onset","pred_offset","type"])

    j=0
    iou = 0.5
    for vio in fileList:
            if os.path.isfile(path+'/'+vio):
                continue
            elif os.path.isdir(path+'/'+vio):
                print(vio)

            if vio[-4:]==".zip":
                continue
            # if vio!="wy_f1_aligned":
            #     continue

            if name == "samm":
                pp = SAMM.draw_roiline19(path , vio , 6, -4,7)  
                pp=pp*7
            elif name == "cas":
                # if vio != "dzw_h3_aligned":
                #     continue
                # if vio != "dzw_s3_aligned":
                #     continue
                # if vio!='16_0101disgustingteeth':
                #     continue
                print("now vio:{}".format(vio))
                #draw_roiwinflow(path, vio, 0, -4, 1)
                pp = CAS.draw_roiline19(path , vio , 0, -4,1)
            pp = pp.tolist()
            pp.sort()
            print(pp)
            for i,interval in enumerate(pp):
                data.loc[i+j,'vid'] = vio[:-8] #去掉 _aligned
                # data.loc[i + j, 'vid'] = vio

                data.loc[i+j,'onset'] = interval[0]
                data.loc[i+j,'offset'] = interval[1]

                # data.loc[i+j,'pred_onset'] = interval[0]
                # data.loc[i+j,'pred_offset'] = interval[1]

                if (interval[1]-interval[0])/fps > iou:
                    data.loc[i+j,'type'] = 'mae'
                else:
                    data.loc[i+j,'type'] = 'me'
            j=i+j+1
    data.to_csv(name+'_pred'+".csv",index=False)

def main():
    # path_samm ='E:/MEGC2023 test/openface_sammalign'  #'/space/qwf/sammtest/SAMM_data/'
    # detect_expression(path_samm,'samm',200)
    path_cas ='E:/MEGC2023 test/openface_cas3align' #'E:/openfaceprocess_22test/cas3'#'D:/graduate/medata/CAS_Test_cropped' #  #'/space/qwf/cas3test/CAS_data/'
    detect_expression(path_cas,'cas',30)
    # pathcas2='D:/graduate/medata/casmesqr/rawpic/s16'
    # detect_expression(pathcas2, 'cas', 30)



if __name__ == '__main__':
    main()
