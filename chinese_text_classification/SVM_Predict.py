#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@version: python3.6
@author: XiangguoSun
@contact: sunxiangguodut@qq.com
@file: NBayes_Predict.py
@time: 2018/1/23 16:12
@software: PyCharm
"""


from sklearn import metrics
from sklearn.svm import SVC
from Tools import readbunchobj
import datetime

def metrics_result(actual, predict):
    print('精度:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
    print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))

if __name__ == '__main__':
    begin_all = datetime.datetime.now()
    # 导入训练集
    trainpath = "train_word_bag/tfdifspace.dat"
    train_set = readbunchobj(trainpath)

    # 导入测试集
    testpath = "test_word_bag/testspace.dat"
    test_set = readbunchobj(testpath)

    begin = datetime.datetime.now()
    svclf = SVC()#defaultwith 'rbf'
    svclf.fit(train_set.tdm, train_set.label)
    # 预测分类结果
    predicted = svclf.predict(test_set.tdm)
    end = datetime.datetime.now()

    result = {}
    for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
        if flabel not in result:
            stat = {"all": 0, "success": 0}
            result[flabel] = stat
        result[flabel]["all"] += 1;
        if flabel == expct_cate:
            result[flabel]["success"] += 1;
            print(file_name, ": Actual:", flabel, " -->expected:", expct_cate)
        else:
            print("Error", file_name, ": Actual:", flabel, " -->expected:", expct_cate)
    print("预测完毕!!!")

    # 计算分类精度：
    metrics_result(test_set.label, predicted)
    end_all = datetime.datetime.now()
    print("Elaspsed time = " + str(((end-begin).microseconds)/1000) + "ms")
    print("Elaspsed all time = " + str((end_all-begin_all).seconds) + "s")

    print("\nAll type precision:")
    for key in result.keys():
        info = result[key]
        success_rate = round(info["success"] / float(info["all"]), 4) * 100
        print(key + ":\t" + str(success_rate) + "%")
