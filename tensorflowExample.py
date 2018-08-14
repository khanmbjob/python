# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 03:39:44 2018

@author: DELL
"""

import struct
from tensorflow.core.example import example_pb2

with open(output_filename, 'wb') as writer:
  body = 'body'
  title = 'title'
    
  tf_example = example_pb2.Example()
  tf_example.features.feature['article'].bytes_list.value.extend([body])
  tf_example.features.feature['abstract'].bytes_list.value.extend([title])
  tf_example_str = tf_example.SerializeToString()
  str_len = len(tf_example_str)
  writer.write(struct.pack('q', str_len))
  writer.write(struct.pack('%ds' % str_len, tf_example_str))