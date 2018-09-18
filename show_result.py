import os
import glob
import json
dd = []
for x in glob.glob('/home/dsl/all_check/aichallenger/ai_chanellger/step3_c/*.json'):
    data = json.loads(open(x).read())
    print(data['cls_name'],data['best_accuracy'])
    dd.append(data['best_accuracy'])
print(sum(dd)/len(dd))