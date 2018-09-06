#coding = utf-8
import shutil
import os
import glob
cls_names = ['番茄', '马铃薯', '葡萄', '樱桃', '苹果', '玉米', '辣椒', '草莓','柑桔','桃子']

root = 'D:/deep_learn_data/ai_chellenger'

def gen_data_step1():
    dr = os.path.join(root,'class_plant_step1','valid')
    tran_set = os.path.join(root,'trainingset')
    test_set = os.path.join(root, 'validationset')
    for x in glob.glob(os.path.join(test_set,'*','*/*.*')):
        for k in cls_names:
            if k in x:
                dt = os.path.join(dr, k)
                if not os.path.exists(dt):
                    os.mkdir(dt)
                shutil.copy(x,dt)


    for x in glob.glob(os.path.join(test_set,'*','*/*/*.*')):
        for k in cls_names:
            if k in x:
                dt = os.path.join(dr, k)
                if not os.path.exists(dt):
                    os.mkdir(dt)
                shutil.copy(x, dt)
def gen_data_step2():
    dr = os.path.join(root,'step2','train')
    tran_set = os.path.join(root,'trainingset','AgriculturalDisease_trainingset')
    test_set = os.path.join(root, 'validationset','AgriculturalDisease_validationset')
    for s in os.listdir(tran_set):
        print(s)
        for cls in cls_names:
            if cls in s:
                dr1 = os.path.join(dr, cls)
                if not os.path.exists(dr1):
                    os.mkdir(dr1)
        n_dr = os.path.join(dr1,s)
        if not os.path.exists(n_dr):
            os.mkdir(n_dr)
        for k in glob.glob(os.path.join(tran_set,s,'*.*')):
            shutil.copy(k, n_dr)
        for k in glob.glob(os.path.join(tran_set, s, '*','*.*')):
            shutil.copy(k, n_dr)

    dr = os.path.join(root, 'step2', 'valid')
    for s in os.listdir(test_set):
        for cls in cls_names:
            if cls in s:
                dr1 = os.path.join(dr, cls)
                if not os.path.exists(dr1):
                    os.mkdir(dr1)
        n_dr = os.path.join(dr1, s)
        if not os.path.exists(n_dr):
            os.mkdir(n_dr)
        for k in glob.glob(os.path.join(test_set, s, '*.*')):
            shutil.copy(k, n_dr)
        for k in glob.glob(os.path.join(test_set, s, '*','*.*')):
            shutil.copy(k, n_dr)
gen_data_step2()
