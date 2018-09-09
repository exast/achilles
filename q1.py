#Loading Libraries:
import pandas as pd

#Loading Database:
db = pd.read_csv("DataChallenge.csv", sep=",") 


"""
1. Some companies declare they have zero employees (FInfo_NumberOfEmployees),
as they are made up of the company director alone.
What is the mean annual turnover among this subset? 
"""

#Filtering the data to get only the companies with zero employees:
zeroemployees = db[db["FInfo_NumberOfEmployees"] == 0]

#Quick Check at the data:
print(zeroemployees)

#As there are only 8 occurrences of zero employees in the dataset
#It is possible to print each of the values of the AnnualTurnover:
print(zeroemployees["FInfo_AnnualTurnover"])

#One of the companies has an AnnualTurnover of 0 pounds,
#which is odd, but possible. Also, a describe() gives a better 
#understanding of the data:
print(zeroemployees["FInfo_AnnualTurnover"].describe())

#For a more explicit way of checking the Mean:
print('£{:,.2f}'.format(zeroemployees["FInfo_AnnualTurnover"].mean()))

#To get the Mean Annual Turnover for the companies with 
#Turnover values above 0, the following code is used:
print('£{:,.2f}'.format(zeroemployees[zeroemployees["FInfo_AnnualTurnover"] > 0]  
                    ["FInfo_AnnualTurnover"].mean()))  


"""Considering missing values as zero employees"""
#The following code is intended to consider the missing values (NaN) in the
#field "FInfo_NumberOfEmployees" as zero.

#Filtering the data to get the companies with zero/NaN employees:
zeroandnan = db[
        (db["FInfo_NumberOfEmployees"] == 0) 
         | 
        (db["FInfo_NumberOfEmployees"].isnull())]

#Quick Check at the data
print(zeroandnan)
print(zeroandnan["FInfo_AnnualTurnover"].describe().apply(lambda x: format(x,'f')))

#It is possible to note that there many occurrences of NaN values in Turnover:
print(zeroandnan["FInfo_AnnualTurnover"]) 

#Finally, to get the mean Annual Turnover for the companies with zero
#or Nan values for "FInfo_NumberOfEmployees" (Considering turnover of £0):
print('£{:,.2f}'.format(zeroandnan["FInfo_AnnualTurnover"].mean()))


