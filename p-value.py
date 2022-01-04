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

weather_no = "results/weather/weather_test-21-12-24-17-47-55-noDrop-21-12-24-17-47-58/weather_test-21-12-24-17-47-55-noDrop-21-12-24-17-47-58-noDrop-21-12-24--17-47-58.csv"
weather_gaus_05 = "results/weather/weather_test_drop-21-12-28-05-18-03/weather_test_drop-21-12-28-05-18-03-gaussianDropout-21-12-28-05-18-06/weather_test_drop-21-12-28-05-18-03-gaussianDropout-21-12-28-05-18-06-gaussianDropout-21-12-28--05-18-06.csv"
weather_dropcon_05 = "results/weather/weather_test_drop-21-12-28-05-18-03/weather_test_drop-21-12-28-05-18-03-dropConnect-21-12-28-05-18-04/weather_test_drop-21-12-28-05-18-03-dropConnect-21-12-28-05-18-04-dropConnect-21-12-28--05-18-04.csv"
weather_adv_0_0001= 'results/weather/weather_test_Adrop-21-12-29-05-56-11/weather_test_Adrop-21-12-29-05-56-11-advancedDropout-21-12-29-05-56-12/weather_test_Adrop-21-12-29-05-56-11-advancedDropout-21-12-29-05-56-12-advancedDropout-21-12-29--05-56-12.csv'
weather_simpl_05 = "results/weather/weather_test_drop-21-12-28-05-18-03/weather_test_drop-21-12-28-05-18-03-simpleDropout-21-12-28-05-18-07/weather_test_drop-21-12-28-05-18-03-simpleDropout-21-12-28-05-18-07-simpleDropout-21-12-28--05-18-07.csv"

bng_no = "results/bng/bng_test_noDrop/bng_test-21-12-30-02-40-58-noDrop-21-12-30-02-40-59/bng_test-21-12-30-02-40-58-noDrop-21-12-30-02-40-59-noDrop-21-12-30--02-40-59.csv"
bng_gaus_05 = "results/bng/bng_test_drop-21-12-29-05-59-24/bng_test_drop-21-12-29-05-59-24-gaussianDropout-21-12-29-05-59-27/bng_test_drop-21-12-29-05-59-24-gaussianDropout-21-12-29-05-59-27-gaussianDropout-21-12-29--05-59-27.csv"
bng_dropcon_05 = "results/bng/bng_test_drop-21-12-29-05-59-24/bng_test_drop-21-12-29-05-59-24-dropConnect-21-12-29-05-59-25/bng_test_drop-21-12-29-05-59-24-dropConnect-21-12-29-05-59-25-dropConnect-21-12-29--05-59-25.csv"
bng_adv_0_0001= "results/bng/bng_test_Adrop-21-12-29-06-00-49/bng_test_Adrop-21-12-29-06-00-49-advancedDropout-21-12-29-06-00-50/bng_test_Adrop-21-12-29-06-00-49-advancedDropout-21-12-29-06-00-50-advancedDropout-21-12-29--06-00-50.csv"
bng_simpl_05 = "results/bng/bng_test_drop-21-12-29-05-59-24/bng_test_drop-21-12-29-05-59-24-simpleDropout-21-12-29-05-59-28/bng_test_drop-21-12-29-05-59-24-simpleDropout-21-12-29-05-59-28-simpleDropout-21-12-29--05-59-28.csv"

boston_drop_pathes = [boston_simpl_05, boston_dropcon_05, boston_gaus_05, boston_adv_0_0001]
callif_drop_pathes = [cal_dropcon_05, cal_gaus_05, cal_simpl_05, cal_adv_0_0001]
weather_drop_pathes = [weather_dropcon_05, weather_gaus_05, weather_simpl_05, weather_adv_0_0001]
bng_drop_pathes = [bng_simpl_05, bng_dropcon_05, bng_adv_0_0001, bng_gaus_05]

datasets = [boston_drop_pathes, callif_drop_pathes, weather_drop_pathes, bng_drop_pathes]

for dataset_path in datasets:
    if dataset_path is boston_drop_pathes :
        df = pd.read_csv(boston_no)
        np_no = df['R^2_test'].to_numpy()
    elif dataset_path is callif_drop_pathes:
        df = pd.read_csv(cal_no)
        np_no = df['R^2_test'].to_numpy()
    elif dataset_path is weather_drop_pathes:
        df = pd.read_csv(weather_no)
        np_no = df['R^2_test'].to_numpy()
    elif dataset_path is bng_drop_pathes:
        df = pd.read_csv(bng_no)
        np_no = df['R^2_test'].to_numpy()
    for path in dataset_path:

        df = pd.read_csv(path)
        np_with_drop = df['R^2_test'].to_numpy()
        p_value = ttest_ind(np_no, np_with_drop)
        print(f"{namestr(path,globals())} : {p_value}")

"""
['boston_simpl_05', 'path'] : Ttest_indResult(statistic=-1.8632591916084953, pvalue=0.02245510722733469)
['boston_dropcon_05', 'path'] : Ttest_indResult(statistic=114.45588032310881, pvalue=0.0)
['boston_gaus_05', 'path'] : Ttest_indResult(statistic=29.603022999083855, pvalue=1.0868336306341836e-184)
['boston_adv_0_0001', 'path'] : Ttest_indResult(statistic=-20.51797206242277, pvalue=1.1318328402843799e-91)

['cal_dropcon_05', 'path'] : Ttest_indResult(statistic=158.66335989499055, pvalue=0.0)
['cal_gaus_05', 'path'] : Ttest_indResult(statistic=49.000374455344335, pvalue=0.0)
['cal_simpl_05', 'path'] : Ttest_indResult(statistic=9.454806823573067, pvalue=3.964820406583049e-21)
['cal_adv_0_0001', 'path'] : Ttest_indResult(statistic=13.803473197450888, pvalue=6.008371195564271e-43)

['weather_dropcon_05', 'path'] : Ttest_indResult(statistic=36.96722710362607, pvalue=1.6472285017418782e-280)
['weather_gaus_05', 'path'] : Ttest_indResult(statistic=78.21163614344583, pvalue=0.0)
['weather_simpl_05', 'path'] : Ttest_indResult(statistic=-22.472828820789037, pvalue=3.754537438891114e-109)
['weather_adv_0_0001', 'path'] : Ttest_indResult(statistic=35.478671791896204, pvalue=8.552538354803772e-260)

['bng_simpl_05', 'path'] : Ttest_indResult(statistic=-119.91579702913332, pvalue=0.0)
['bng_dropcon_05', 'path'] : Ttest_indResult(statistic=303.50822879413414, pvalue=0.0)
['bng_adv_0_0001', 'path'] : Ttest_indResult(statistic=22.159008314849196, pvalue=3.0025129294882315e-106)
['bng_gaus_05', 'path'] : Ttest_indResult(statistic=-27.136686389236175, pvalue=2.70244846604008e-156)
"""