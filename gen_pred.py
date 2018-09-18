import os
import json
import glob
dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/pred/'
l1 = glob.glob(os.path.join(dr,'*','*','*.*'))
l2 = glob.glob(os.path.join(dr,'*','*','*','*.*'))

l1.extend(l2)
lbs = json.loads(open('label.json').read())

def gen_valid():
    org_map = dict()
    org = json.loads(open(
        '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/valid/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json').read())
    for s in org:
        org_map[s['image_id']] = s['disease_class']

    print(lbs)

    r = 0
    tt = 0
    for x in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/valid_pred/桃子/桃子健康/*.*'):
        tt+=1
        d = x.split('/')
        index = d[-1]
        if len(d) == 10:
            ll = ''.join(d[-3:-1])
        else:
            ll = d[-2]
        if ll == lbs[str(org_map[index])]:
            r += 1
        else:
            print(ll, lbs[str(org_map[index])])
    print(r / tt)



def gen_pred():
    new_lbs = dict()
    for x in lbs:
        new_lbs[lbs[x]] = x
    print(new_lbs)
    result = []
    for x in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/pred_step2/*/*/*.*'):

        print(x)
        d = x.split('/')
        index = d[-1]
        ll = d[-2]
        result.append({'image_id': index, 'disease_class': int(new_lbs[ll])})
    print(len(result))
    with open('base_line.json', 'w') as f:
        f.write(json.dumps(result))
        f.flush()



gen_pred()