#Loading Libraries:
import pandas as pd
from matplotlib import pyplot as plt

#Loading Database:
db = pd.read_csv("DataChallenge.csv", sep=",") 


"""
3. What is the mean annual turnover among all companies? 
And the median? Could you elaborate on why the two values differ? 
How would you define an unreasonably low/unreasonably high turnover?
How would you visualize this data? 
"""

"""
What is the mean annual turnover among all companies? And the median?
"""
#Quick Check at the data:
print(db['FInfo_AnnualTurnover'].describe().apply(lambda x: format(x, 'f')))

#Finding the mean Annual Turnover among all companies:
print('£{:,.2f}'.format(db['FInfo_AnnualTurnover'].mean())) 
        #Ignores missing NaN values
        
#Finding the median Annual Turnover among all companies: 
print('£{:,.2f}'.format(db['FInfo_AnnualTurnover'].median()))
        #Ignores missing NaN values

#We could also give a deeper look at the data and find if there are some
#missing values in the dataset:
print(len(db[db['FInfo_AnnualTurnover'].isnull()]))
#This code yields a subset of 24 missing values of the Annual Turnover,
#which is only 0.37% of the Dataset, so ignoring it will not affect our study 

#I also looked for any values that might be considered too low or negative, by running:
print(db[db['FInfo_AnnualTurnover']<100]['FInfo_AnnualTurnover'].describe())

#This code shows that there are 35 occurrences of AnnualTurnover below £100,
#with most being £0 and none of them being negative (which is a good sign of
#data hygiene). Even though 35 occurrences is negligible for the size of the
#dataset, it is possible to get the Median and Mean values of the 
#"more significant" values (non-NaN and above £100) with the following code:

relevantturnover = db[db['FInfo_AnnualTurnover']>=100]
print('£{:,.2f}'.format(relevantturnover['FInfo_AnnualTurnover'].mean()))
print('£{:,.2f}'.format(relevantturnover['FInfo_AnnualTurnover'].median()))




"""
Could you elaborate on why the two values differ? 
"""

#Plotting the histogram of Turnovers to see the distribution of the data:

plt.title('Annual Turnover Histogram', fontsize = 15) #Graph Title
plt.xlabel('Annual Turnover [in £]', fontsize = 12) #X-axis label
plt.ylabel('N° of occurrences', fontsize = 12) #Y-axis label
db['FInfo_AnnualTurnover'].hist(bins = 30)
plt.plot()
plt.show()

#Plotting the histogram of Turnovers in Logarithmic Scale:

plt.title('Annual Turnover Histogram', fontsize = 15) #Graph Title
plt.xlabel('Annual Turnover [in £]', fontsize = 12) #X-axis label
plt.ylabel('N° of occurrences (log scale)', fontsize = 12) #Y-axis label
plt.yscale('log') #plotting in log scale
db['FInfo_AnnualTurnover'].hist(bins = 30)
plt.plot()
plt.show()


"""
How would you define an unreasonably low/unreasonably high turnover? 
"""

#Defining Outliers:
turnover = db['FInfo_AnnualTurnover']

q1 = turnover.quantile(q=.25)
q2 = turnover.quantile(q=.50)
q3 = turnover.quantile(q=.75)
iqr =  q3 - q1
lowbound = q1 - (1.5 * iqr)
hibound= q3 + (1.5 * iqr)
print(
      'q1: ','£{:,.2f}'.format(q1),'\n',
      'q2: ','£{:,.2f}'.format(q2),'\n',
      'q3: ','£{:,.2f}'.format(q3),'\n',
      'IQR: ','£{:,.2f}'.format(iqr),'\n',
      'Lower Boundary: ','£{:,.2f}'.format(lowbound),'\n',
      'Upper Boundary: ','£{:,.2f}'.format(hibound),'\n',
      )

#Finding the Standard Deviation of the sample:
print('£{:,.2f}'.format(db['FInfo_AnnualTurnover'].std())) 


#Calculating % of Outliers:
outliers = db[db['FInfo_AnnualTurnover']>hibound]['FInfo_AnnualTurnover'].count()
print('Percentage of Outliers in the data:',"{0:.1%}".format(outliers/len(db)))


#Defining boundaries 99.7%
print('Min:','£{:,.2f}'.format(db['FInfo_AnnualTurnover'].quantile(.0015)))  
print('Max:','£{:,.2f}'.format(db['FInfo_AnnualTurnover'].quantile(.9985)))  
print('Outliers:', 2*(db[db['FInfo_AnnualTurnover']>
                   (db['FInfo_AnnualTurnover'].quantile(.9985))]
                   ['FInfo_AnnualTurnover'].count()))  
  

"""
How would you visualize this data? 
"""
#Defining the NotNull values for the boxplot:
box = db.sort_values('FInfo_AnnualTurnover')[db['FInfo_AnnualTurnover'].notnull()]['FInfo_AnnualTurnover']
box.to_frame()

#Building a boxplot with the upper and lower extremes of 0.15% and 99.85%:
plt.title('BoxPlot of Annual Turnover', fontsize = 15)
plt.xlabel('Extremes of 0.15% and 99.85%', fontsize = 12)
plt.ylabel('Annual Turnover [in £]', fontsize = 12)
plt.boxplot(box, whis=[.15,99.85])
plt.plot()
plt.show()

#Building the same boxplot in Log Scale:  
plt.title('BoxPlot - Turnover in Log Scale', fontsize = 15)
plt.xlabel('Extremes of 0.15% and 99.85%', fontsize = 12) 
plt.ylabel('Annual Turnover [in £] (log scale)', fontsize = 12)  
plt.yscale('log')   
plt.boxplot(box, whis=[.15,99.85])  
plt.plot()  
plt.show()  


#Building the boxplot in Log Scale with full axis:
plt.figure(num=None, figsize=(6, 8), dpi=80, facecolor='w', edgecolor='k')
plt.title('BoxPlot - Turnover in Log Scale', fontsize = 15)
plt.xlabel('Extremes of 0.15% and 99.85%', fontsize = 12)
plt.ylabel('Annual Turnover [in £] (log scale)', fontsize = 12)
plt.boxplot(box, whis=[.15,99.85])
plt.ylim(ymin=1, ymax = 10**11) 
plt.yscale('log')
plt.plot()
plt.show()
