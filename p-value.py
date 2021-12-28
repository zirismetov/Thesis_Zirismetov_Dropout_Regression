from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

boston_no = "results/boston/boston_houses_test_noDrop/boston_houses_test-21-12-25-16-30-06-noDrop-21-12-25-16-30-09-noDrop-21-12-25--16-30-09.csv"
boston_gaus_05 = "results/boston/boston_houses_drop_05/boston_houses_drop-21-12-28-06-40-43-gaussianDropout-21-12-28-06-40-48/boston_houses_drop-21-12-28-06-40-43-gaussianDropout-21-12-28-06-40-48-gaussianDropout-21-12-28--06-40-48.csv"
boston_adv_0_0001 = "results/boston/boston_houses_drop_05/boston_houses_Adrop-21-12-28-07-39-20/boston_houses_Adrop-21-12-28-07-39-20-advancedDropout-21-12-28-07-39-25/boston_houses_Adrop-21-12-28-07-39-20-advancedDropout-21-12-28-07-39-25-advancedDropout-21-12-28--07-39-25.csv"
boston_simpl_05 = "results/boston/boston_houses_drop_05/boston_houses_drop-21-12-28-06-40-43-simpleDropout-21-12-28-06-40-50/boston_houses_drop-21-12-28-06-40-43-simpleDropout-21-12-28-06-40-50-simpleDropout-21-12-28--06-40-50.csv"
boston_dropcon_05 = "results/boston/boston_houses_drop_05/boston_houses_drop-21-12-28-06-40-43-dropConnect-21-12-28-06-40-48/boston_houses_drop-21-12-28-06-40-43-dropConnect-21-12-28-06-40-48-dropConnect-21-12-28--06-40-48.csv"

cal_no = "results/calif/cal_houses_test_noDrop/cal_houses_test-21-12-25-16-29-18-noDrop-21-12-25-16-29-23-noDrop-21-12-25--16-29-23.csv"
cal_dropcon_05 = 'results/calif/cal_houses_test_drop_05/cal_houses_test_drop-21-12-28-08-00-29-dropConnect-21-12-28-08-00-34/cal_houses_test_drop-21-12-28-08-00-29-dropConnect-21-12-28-08-00-34-dropConnect-21-12-28--08-00-34.csv'
cal_gaus_05 = "results/calif/cal_houses_test_drop_05/cal_houses_test_drop-21-12-28-08-00-29-gaussianDropout-21-12-28-08-00-35/cal_houses_test_drop-21-12-28-08-00-29-gaussianDropout-21-12-28-08-00-35-gaussianDropout-21-12-28--08-00-35.csv"
cal_adv_0_0001= "results/calif/cal_houses_test_drop_05/cal_houses_test_Adrop-21-12-28-07-53-44/cal_houses_test_Adrop-21-12-28-07-53-44-advancedDropout-21-12-28-07-53-48/cal_houses_test_Adrop-21-12-28-07-53-44-advancedDropout-21-12-28-07-53-48-advancedDropout-21-12-28--07-53-48.csv"
cal_simpl_05 = "results/calif/cal_houses_test_drop_05/cal_houses_test_drop-21-12-28-08-00-29-simpleDropout-21-12-28-08-00-36/cal_houses_test_drop-21-12-28-08-00-29-simpleDropout-21-12-28-08-00-36-simpleDropout-21-12-28--08-00-36.csv"

boston_drop_pathes = [boston_simpl_05, boston_dropcon_05, boston_gaus_05, boston_adv_0_0001]
callif_drop_pathes = [cal_dropcon_05, cal_gaus_05, cal_simpl_05, cal_adv_0_0001]
datasets = [boston_drop_pathes, callif_drop_pathes]

for dataset_path in datasets:
    if dataset_path is boston_drop_pathes :
        df = pd.read_csv(boston_no)
        np_no = df['R^2_test'].to_numpy()
    elif dataset_path is callif_drop_pathes:
        df = pd.read_csv(cal_no)
        np_no = df['R^2_test'].to_numpy()
    for path in dataset_path:

        df = pd.read_csv(path)
        np_with_drop = df['R^2_test'].to_numpy()
        p_value = ttest_ind(np_no, np_with_drop)
        print(f"{namestr(path,globals())} : {p_value}")

"""
['boston_simpl_05', 'path'] : Ttest_indResult(statistic=-1.8632591916084953, pvalue=0.06245510722733469)
['boston_dropcon_05', 'path'] : Ttest_indResult(statistic=114.45588032310881, pvalue=0.0)
['boston_gaus_05', 'path'] : Ttest_indResult(statistic=29.603022999083855, pvalue=1.0868336306341836e-184)
['boston_adv_0_0001', 'path'] : Ttest_indResult(statistic=-20.51797206242277, pvalue=1.1318328402843799e-91)
['cal_dropcon_05', 'path'] : Ttest_indResult(statistic=158.66335989499055, pvalue=0.0)
['cal_gaus_05', 'path'] : Ttest_indResult(statistic=49.000374455344335, pvalue=0.0)
['cal_simpl_05', 'path'] : Ttest_indResult(statistic=9.454806823573067, pvalue=3.964820406583049e-21)
['cal_adv_0_0001', 'path'] : Ttest_indResult(statistic=13.803473197450888, pvalue=6.008371195564271e-43)
"""