# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 01:12:20 2018

@author: Mohammed Barkath
name: mpg_prediction
description: Predict MPG using a linear regression model
"""

import pandas as pd

from sklearn.externals.joblib import load
from bokeh.plotting import figure
from bokeh.io import curdoc, output_file

mpg_model = load('mpg_linear_regression_model.pkl')
data      = pd.read_csv('mpg_data.csv')

# augment the mpg DataFrame with the prediction
data['prediction'] = mpg_model.predict(data['cyl displ hp weight accel yr'.split()])

s1 = figure()
s1.Scatter(data=data,
            x='mpg', y='prediction', color='origin',
            height=300, width=600,
            title='Fuel efficiency predictions of selected vehicles from 1970-1982',
            tools='hover, box_zoom, lasso_select, save, reset',
            tooltips = [
              ('model','@name'),
              ('HP',  '@hp'),
              ('actual MPG', '@mpg'),
              ('predicted MPG', '@prediction')
            ])

s2 = figure()
s2.Scatter(data=data,
            x='yr', y='mpg', color='origin',
            height=300, width=600,
            title='Fuel efficiency of selected vehicles from 1970-1982',
            tools='hover, box_zoom, save, reset',
            tooltips = [
              ('model','@name'),
              ('HP',  '@hp'),
              ('cyl', '@cyl'),
              ('weight', '@weight')
            ])

curdoc().add_root(s1)
curdoc().add_root(s2)
curdoc().title = "MPG Prediction"