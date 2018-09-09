# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:03:28 2018

@author: rick
"""

print("DB0 LowPrice:",db0["LowPrice"].sum()," (",
      '{:.2f}'.format(100*db0["LowPrice"].sum()/len(db0)),") \n",

      "DB0 MedPrice:",db0["MedPrice"].sum()," (",
      '{:.2f}'.format(100*db0["MedPrice"].sum()/len(db0)),") \n",
      
      "DB0 HighPrice:",db0["HighPrice"].sum()," (",
      '{:.2f}'.format(100*db0["HighPrice"].sum()/len(db0)),") \n",)

print("DB LowPrice:",db["LowPrice"].sum()," (",
      '{:.2f}'.format(100*db["LowPrice"].sum()/len(db)),") \n",

      "DB MedPrice:",db["MedPrice"].sum()," (",
      '{:.2f}'.format(100*db["MedPrice"].sum()/len(db)),") \n",
      
      "DB HighPrice:",db["HighPrice"].sum()," (",
      '{:.2f}'.format(100*db["HighPrice"].sum()/len(db)),") \n",)


print("Terrace LowPrice:",terracedb["LowPrice"].sum()," (",
      '{:.2f}'.format(100*terracedb["LowPrice"].sum()/len(terracedb)),") \n",

      "Terrace MedPrice:",terracedb["MedPrice"].sum()," (",
      '{:.2f}'.format(100*terracedb["MedPrice"].sum()/len(terracedb)),") \n",
      
      "Terrace HighPrice:",terracedb["HighPrice"].sum()," (",
      '{:.2f}'.format(100*terracedb["HighPrice"].sum()/len(terracedb)),") \n",)


print("Detached LowPrice:",detacheddb["LowPrice"].sum()," (",
      '{:.2f}'.format(100*detacheddb["LowPrice"].sum()/len(detacheddb)),") \n",

      "Detached MedPrice:",detacheddb["MedPrice"].sum()," (",
      '{:.2f}'.format(100*detacheddb["MedPrice"].sum()/len(detacheddb)),") \n",
      
      "Detached HighPrice:",detacheddb["HighPrice"].sum()," (",
      '{:.2f}'.format(100*detacheddb["HighPrice"].sum()/len(detacheddb)),") \n",)
