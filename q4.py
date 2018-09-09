#Loading Libraries:
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Loading Database:
db = pd.read_csv("DataChallenge.csv", sep=",") 

"""
4. Let’s look at companies that have meaningfully large turnover
(greater than 1000 pounds) and have had some accidents causing lost time
(HAS_LostTimeFrequency). Do you see any correlation between the two?
How would you visualize the problem? How would you quantify the correlation?
"""

largewaccidents = db[
                    (db['FInfo_AnnualTurnover'] >1000)
                    &
                    (db['HAS_LostTimeFrequency'] > 0)][
                            ['FInfo_AnnualTurnover','HAS_LostTimeFrequency']]

print(largewaccidents['HAS_LostTimeFrequency'].describe())

#Histogram of LostTimeFrequency
plt.title('LostTimeFrequency Histogram', fontsize = 15)
plt.xlabel('LostTimeFrequency', fontsize = 12)
plt.ylabel('N° of occurrences (log scale)', fontsize = 12)
plt.yscale('log') 
largewaccidents['HAS_LostTimeFrequency'].hist(bins = 30)
plt.plot()
plt.show()


#Scatter Plot:
plt.figure(num=None, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(largewaccidents['HAS_LostTimeFrequency'],
            largewaccidents['FInfo_AnnualTurnover'])
plt.title('Scatter Plot 1: Correlation Study', fontsize = 15)
plt.xlabel('LostTimeFrequency', fontsize = 12)
plt.ylabel('Annual Turnover [in £]', fontsize = 12)
plt.plot()
plt.show()


#Logarithmic Scatter Plot:
plt.figure(num=None, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(largewaccidents['HAS_LostTimeFrequency'],
            largewaccidents['FInfo_AnnualTurnover'])
plt.title('Scatter Plot 2: Logarithmic scale', fontsize = 15)
plt.xlabel('LostTimeFrequency', fontsize = 12)
plt.ylabel('Annual Turnover [in £]', fontsize = 12)
plt.xscale('log') 
plt.yscale('log') 
plt.plot()
plt.show()


####Barplot:
plt.bar(range (len(largewaccidents)),
        largewaccidents.sort_values('FInfo_AnnualTurnover')
        ['HAS_LostTimeFrequency'])
plt.title('Bar Chart (Log Scale)', fontsize = 15)
plt.xlabel('Ranking of Annual Turnover', fontsize = 12)
plt.ylabel('LostTimeFrequency', fontsize = 12)
plt.yscale('log')
plt.plot()
plt.show() 



"""Quantifying the correlation:"""

largewaccidents.corr(method='pearson').iloc[1,0]  #Out: 0.2966
largewaccidents.corr(method='spearman').iloc[1,0] #Out: -0.5636


l0 = largewaccidents

l1 = l0[(l0['FInfo_AnnualTurnover'] >= 
         l0['FInfo_AnnualTurnover'].sort_values().iloc[10])]
[['FInfo_AnnualTurnover','HAS_LostTimeFrequency']]

l2 = l1[(l1['FInfo_AnnualTurnover'] <= 
         l0['FInfo_AnnualTurnover'].sort_values().iloc[-10])]
[['FInfo_AnnualTurnover','HAS_LostTimeFrequency']]

l3 = l2[(l2['HAS_LostTimeFrequency'] >= 
         l0['HAS_LostTimeFrequency'].sort_values().iloc[10])]
[['FInfo_AnnualTurnover','HAS_LostTimeFrequency']]

l4 = l3[(l3['HAS_LostTimeFrequency'] <= 
         l0['HAS_LostTimeFrequency'].sort_values().iloc[-10])]
[['FInfo_AnnualTurnover','HAS_LostTimeFrequency']]

LWANormal = l4

print(LWANormal.corr(method='pearson').iloc[1,0])
print(LWANormal.corr(method='spearman').iloc[1,0])

