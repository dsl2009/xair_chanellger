#coding = utf-8
import shutil
import os
import glob
import re
import json
cls_names = ['番茄', '马铃薯', '葡萄', '樱桃', '苹果', '玉米', '辣椒', '草莓','柑桔','桃子']


root = 'D:/deep_learn_data/ai_chellenger'
js_valid = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/valid/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json'
ig_dr_valid = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/valid/AgriculturalDisease_validationset/images'

js_train = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/train/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json'
ig_dr_train = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/train/AgriculturalDisease_trainingset/images'

def gen_step1_label():
    regex_str = ".*?([\u4E00-\u9FA5])"

    step1 = dict()
    for x in open('total_labels.txt').readlines():
        d = x.replace('\n','').split('\t')
        label_id = d[0]
        match_obj = re.match(regex_str, d[1])
        if '樱桃' not in d[1]:
            r = re.findall(r'[\u4e00-\u9fa5]',  d[1].replace('桃疮痂','桃子疮痂').replace('桃健康','桃子健康'))
        else:
            r = re.findall(r'[\u4e00-\u9fa5]', d[1])
        china_name = ''.join(r)

        step1[label_id] = china_name
    with open('label.json','w') as f:
        f.write(json.dumps(step1))
        f.flush()

def gen_base_line():
    js = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/valid/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json'
    ig_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/valid/AgriculturalDisease_validationset/images'
    am_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/base_line_valid'
    label = json.loads(open('label.json').read())
    for x in json.loads(open(js).read()):
        cls_id = x['disease_class']
        img_id = x['image_id']
        ndr = os.path.join(am_dr,str(cls_id))
        if not os.path.exists(ndr):
            os.makedirs(ndr)
        shutil.copy(os.path.join(ig_dr,img_id),ndr)

def gen_step1():
    js = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/train/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json'
    ig_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/train/AgriculturalDisease_trainingset/images'
    am_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/step1'
    label = json.loads(open('label.json').read())

    for x in json.loads(open(js).read()):
        cls_id = x['disease_class']
        img_id = x['image_id']
        real_name = label[str(cls_id)]
        for d in cls_names:
            if d in real_name:
                ndr = os.path.join(am_dr,d)
                if not os.path.exists(ndr):
                    os.mkdir(ndr)
                shutil.copy(os.path.join(ig_dr,img_id),ndr)

def gen_step2():
    js = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/valid/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json'
    ig_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/valid/AgriculturalDisease_validationset/images'
    am_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/step2_valid'
    label = json.loads(open('label.json').read())
    for x in json.loads(open(js).read()):
        cls_id = x['disease_class']
        img_id = x['image_id']
        real_name = label[str(cls_id)]
        ids = 0
        for d in cls_names:

            if d in real_name:
                d1 = real_name.replace('一般','').replace('严重', '')

                ndr = os.path.join(am_dr,d,d1)
                if not os.path.exists(ndr):
                    os.makedirs(ndr)
                shutil.copy(os.path.join(ig_dr,img_id),ndr)
                ids = 1
        if ids == 0:
            print(real_name)
def gen_step3():
    js = js_train
    ig_dr = ig_dr_train
    am_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/nn_new/step2_train'
    label = json.loads(open('label.json').read())

    for x in json.loads(open(js).read()):
        cls_id = x['disease_class']
        img_id = x['image_id']
        real_name = label[str(cls_id)]
        for d in cls_names:
            if d in real_name:
                ndr = os.path.join(am_dr, d,real_name)
                '''
                 d1 = real_name.replace('一般','').replace('严重', '')
                if '一般' in real_name:
                    ndr = os.path.join(am_dr,d,d1,'一般')
                elif '严重' in real_name:
                    ndr = os.path.join(am_dr, d, d1, '严重')
                else:
                    break
                '''

                if not os.path.exists(ndr):
                    os.makedirs(ndr)
                shutil.copy(os.path.join(ig_dr,img_id),ndr)


gen_step3()