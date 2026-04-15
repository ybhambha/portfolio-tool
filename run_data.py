# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:15:39 2026

@author: bhamb
"""

from src.data import build_data_pipeline

data = build_data_pipeline()
print(data["prices"].tail())
print(data["log_returns"].tail())