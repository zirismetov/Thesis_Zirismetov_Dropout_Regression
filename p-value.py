from scipy.stats import ttest_ind
import numpy as np
import pandas as pd

file_no_drop_path = 'results/boston_houses_test_0_001/boston_houses_test-21-12-25-16-30-06-noDrop-21-12-25-16-30-09/boston_houses_test-21-12-25-16-30-06-noDrop-21-12-25-16-30-09-noDrop-21-12-25--16-30-09.csv'
file_with_drop_path = 'results/boston_houses_drop_DC-21-12-27-05-03-34/boston_houses_drop_DC-21-12-27-05-03-34-simpleDropout-21-12-27-05-03-37/boston_houses_drop_DC-21-12-27-05-03-34-simpleDropout-21-12-27-05-03-37-simpleDropout-21-12-27--05-03-37.csv'

df = pd.read_csv(file_no_drop_path)
np_no = df['R^2_test'].to_numpy()

df = pd.read_csv(file_with_drop_path)
np_with = df['R^2_test'].to_numpy()

p_value = ttest_ind(np_no, np_with)
print(p_value)