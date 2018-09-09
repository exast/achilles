#Loading Libraries:
import pandas as pd

#Loading Database:
db = pd.read_csv("DataChallenge.csv", sep=",") 

"""
2. Some companies have a parent company (UltimateParentExists). 
What is the fourth most common currency (UltimateParent_Currency) among them?
"""

#Using unique() on "UltimateParentExists" column 
#to check which entries are there in the field:
print(db['UltimateParentExists'].unique())


#Checking how many 'nan' values are present in the subset:
print(db[db['UltimateParentExists'].isnull()])

#47 occurrences of NaN values in the field "UltimateParentExists"
#(out of 6466), represents only 0.7% of the total. 
#Therefore, these values will be ignored by now,
#as they probably mean that there is no Parent Company.

#Creating a subset of the companies with Ultimate Parents
subsidiaries = db[db['UltimateParentExists']=='Yes'] 

#As usual, a quick look at the data we are working with:
print(subsidiaries)
print(subsidiaries.describe())
print(subsidiaries['UltimateParent_Currency'].unique())

#There are many different currencies in the dataset, 
#again with some being NaN values.
#So, it is necessary to check if this is a common value:
print(len(subsidiaries[subsidiaries['UltimateParent_Currency'].isnull()]))

#The code above returns 799 rows, which can be considered a large part
#of the subset. There are a few possibilities here, some of them being:
#this information is actually missing (the questionnaires were not filled
#properly) or it can be the case that the team that fills the questionnaires
#considers a particular Currency (i.e: US Dollar, Sterling Pound or any other)
#as a Default, so it does not always fill out this field. Proceeding:
    
#Finding a list of the number of occurences of unique values in the subset:
print(subsidiaries['UltimateParent_Currency'].value_counts())


#This result shows that the Swedish Krona is the fourth most currency among 
#this subset with 23 occurrences, behind the Sterling Pound (1291), Euro (278)
#and US Dollar (172). This means that the information behind the NaN values
#(799 occurrences) in this subset is certainly relevant and any important 
#decision should not be based in the values above alone, without considering
#that there are many missing values. Also, there is a possibility of the
#US Dollar being the 4th most common currency in the subset (172 occurrences)
#if the Missing values refer to some other currency unknown here.
#(even though it does not seem to be plausible here)













