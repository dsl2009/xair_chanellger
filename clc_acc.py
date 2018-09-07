import glob
import os


def clc_total():
    l1 = glob.glob(
        '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/org/valid/AgriculturalDisease_validationset/*/*.*')
    l2 = glob.glob(
        '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/org/valid/AgriculturalDisease_validationset/*/*/*.*')
    org = dict()
    for x in l1:
        d = x.split('/')
        org[d[-1]] = d[-2]
    for x in l2:
        d = x.split('/')
        org[d[-1]] = '-'.join(d[-3:-1])
    l1 = glob.glob(
        '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/valid_test/*/*/*.*')
    l2 = glob.glob(
        '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/valid_test/*/*/*/*.*')
    pred = dict()
    for x in l1:
        d = x.split('/')
        pred[d[-1]] = d[-2]
    for x in l2:
        d = x.split('/')
        pred[d[-1]] = '-'.join(d[-3:-1])
    tt = 0
    r = 0
    for dx in org:

        if org[dx] == pred[dx]:
            r += 1
        else:
            print(org[dx], pred[dx])
        tt += 1
    print(r, tt)
    print(r / tt)


def clc_total_fen():
    l1 = glob.glob(
        '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/valid_test/番茄/番茄 白粉病/*/*.*')

    org = dict()

    for x in l1:
        d = x.split('/')
        org[d[-1]] = '-'.join(d[-3:-1])
    l1 = glob.glob(
        '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/org/valid/AgriculturalDisease_validationset/番茄 白粉病/*/*.*')

    pred = dict()
    for x in l1:
        d = x.split('/')
        pred[d[-1]] = '-'.join(d[-3:-1])
    tt = 0
    r = 0
    for dx in org:

        if org[dx] == pred[dx]:
            r += 1
        else:
            print(org[dx], pred[dx])
        tt += 1
    print(r, tt)
    print(r / tt)
clc_total_fen()