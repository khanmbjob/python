# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 18:20:33 2018

@author: DELL
"""

import matplotlib.pyplot as plt
import seaborn as sns

import warnings 
import pandas as pd
import os


print(os.getcwd())
os.chdir('C:\\Users\\DELL\\Desktop\\Python')

stats2 = pd.read_csv('MasterData.csv')

warnings.filterwarnings('ignore')
visl = sns.distplot(stats2["Weight"],bins=30) 
plt1 = sns.boxplot(data=stats2, x="Porject ID", y="Weight")

