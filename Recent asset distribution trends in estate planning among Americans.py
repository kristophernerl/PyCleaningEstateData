import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split 

os.chdir("C://Users/krist/Downloads/h20sas")

#COLLECTING & MERGING DATA
h20b_r = pd.read_sas('h20b_r.sas7bdat') #convert Demographics sas file to csv
h20b_r.to_csv('h20b_r.csv', index=False)
h20q_h = pd.read_sas('h20q_h.sas7bdat') #convert Assets and Income sas file to csv
h20q_h.to_csv('h20q_h.csv', index=False)
h20t_r = pd.read_sas('h20t_r.sas7bdat') #convert Wills and Life Insurance sas file to csv
h20t_r.to_csv('h20t_r.csv', index=False)

#outer join Demographics and Assets and Income 
merged_df = h20q_h.merge(h20b_r, how='outer', left_on=["HHID", "RSUBHH", "QSUBHH"], right_on=["HHID", "RSUBHH", "QSUBHH"]) 

#outer join previous merge and Wills and Life Insurance 
merged_df2 = h20t_r.merge(merged_df, how='outer', left_on=["HHID"], right_on=["HHID"])
merged_df2.to_csv('joinedall.csv', index = False)  

df=merged_df2[[('HHID', 'RSUBHH', 'QSUBHH', 'RPN_SP', 'RT001', 'RT003', 'RT004M1',                
               'RT004M2', 'RT004M3', 'RT004M4', 'RT005', 'RT006', 'RT008', 'RT011', 
               'RT013', 'RQ020', 'RB017M', 'RB020', 'RB022', 'RB089M1M', 'RQ454')]] #reduced dataset

df.isna().sum().sum() #19995268 missing values across the merged dataframe.

df["RT001"].value_counts()

df["RT003"].value_counts()

df["RT004M1"].value_counts()

df["RT004M2"].value_counts()

df["RT004M3"].value_counts()

df["RT004M4"].value_counts()

df["RT005"].value_counts()

df["RT006"].value_counts()

df["RT008"].value_counts()

df["RT011"].value_counts()

df["RT013"].describe()

df["RQ020"].describe()

df["RB017M"].value_counts()

df["RB020"].value_counts()

df["RB022"].value_counts()

df["RB089M1M"].value_counts()

df["RQ454"].describe()


df.isna().sum().sum() #19995268 missing values across the merged dataframe.

print(df[df.duplicated()])

#INVESTIGATING FOR ISSUES
df.drop_duplicates(keep="first",inplace=True)

print(df[df.duplicated()])


df = pd.read_csv("compressed data2.csv")

print(df[df.duplicated()])

#INVESTIGATING FOR ISSUES
df.drop_duplicates(keep="first",inplace=True)

print(df[df.duplicated()])

#number of irrelevent inputs
df[df == -8].count()

df[df == 8].count()

df[df == 9].count()

df[df == 98].count()

df[df == 99].count()

subset = df.iloc[:, 4:21]

subset.nunique(dropna=True)

subset.dtypes.value_counts()

# Box Plot RT013
filtered_RT013 = subset["RT013"][~np.isnan(subset["RT013"])]
fig, ax = plt.subplots(figsize=(20,6))
sns.boxplot(x=filtered_RT013)
plt.title('Box Plot: total face value of life insurance policies', fontsize=15)
plt.xlabel('(hundreds of millions of dollars)', fontsize=14)
plt.show()

filtered_RT013.describe()
filtered_RT013[filtered_RT013 > 150000000].describe()

fig, ax = plt.subplots(figsize=(20,6))
sns.boxplot(x=filtered_RT013[filtered_RT013 < 150000001])
plt.title('Box Plot: total face value of life insurance policies', fontsize=15)
plt.xlabel('(hundreds of millions of dollars)', fontsize=14)
ax.set_xlim(0, 150000001)
plt.show()

# Box Plot RQ020
filtered_RQ020 = subset["RQ020"][~np.isnan(subset["RQ020"])]
filtered_RQ020.describe()

filtered_RQ020[filtered_RQ020 > 2500000].describe()

fig, ax = plt.subplots(figsize=(20,6))
sns.boxplot(x=filtered_RT013[filtered_RT013 < 2500001])
plt.title('Box Plot: wage and salary income in the Last Calendar Year', fontsize=15)
plt.xlabel('(millions of dollars)', fontsize=14)
ax.set_xlim(0, 2500001)
plt.show()

# Box Plot RQ454 
filtered_RQ454 = subset["RQ454"][~np.isnan(subset["RQ454"])]
filtered_RQ454.describe()

filtered_RQ454[filtered_RQ454 > 201000].describe()

fig, ax = plt.subplots(figsize=(20,6))
sns.boxplot(x=filtered_RQ454[filtered_RQ454 < 201001])
plt.title('Box Plot: Total amount donated to Charity', fontsize=15)
plt.xlabel('(dollars)', fontsize=14)
ax.set_xlim(0, 201001)
plt.show()

#Distribution of RT013
histRT013 = filtered_RT013[filtered_RT013 < 150000001].plot(kind='hist', title='Distribution of Filtered RT013')
filtered_RT013[filtered_RT013 < 150000001].skew() #extremely skewed 

histRQ020 = filtered_RQ020[filtered_RQ020 < 2500001].plot(kind='hist', title='Distribution of Filtered RQ020')
filtered_RQ020[filtered_RQ020 < 2500001].skew() #extremely skewed 

histRQ454 = filtered_RQ454[filtered_RQ454 < 201001].plot(kind='hist', title='Distribution of Filtered RQ454')
filtered_RQ454[filtered_RQ454 < 201001].skew() #extremely skewed 


# Count of missing values of each column
subset.isna().sum()

msno.bar(subset)

subset.isnull().values.ravel().sum()

len(subset) - len(subset.dropna(how='all'))#entire rows of non-id attributes are null


#ACTUAL CLEANING
#replacing Person Index Number with "1"
clean = df
clean['RT004M1'] = np.where(clean['RT004M1'].between(3,39), 1, clean['RT004M1'])
clean['RT004M2'] = np.where(clean['RT004M2'].between(3,39), 1, clean['RT004M2'])
clean['RT004M3'] = np.where(clean['RT004M3'].between(3,39), 1, clean['RT004M3'])
clean['RT004M4'] = np.where(clean['RT004M4'].between(3,39), 1, clean['RT004M4'])
clean['RT004M1'] = np.where(clean['RT004M1'] == 93, 2, clean['RT004M1'])
clean['RT004M2'] = np.where(clean['RT004M2'] == 93, 2, clean['RT004M2'])
clean['RT004M3'] = np.where(clean['RT004M3'] == 93, 2, clean['RT004M3'])
clean['RT004M4'] = np.where(clean['RT004M4'] == 93, 2, clean['RT004M4'])
clean['RT004M1'] = np.where(clean['RT004M1'] == 96, 3, clean['RT004M1'])
clean['RT004M2'] = np.where(clean['RT004M2'] == 96, 3, clean['RT004M2'])
clean['RT004M3'] = np.where(clean['RT004M3'] == 96, 3, clean['RT004M3'])
clean['RT004M4'] = np.where(clean['RT004M4'] == 96, 3, clean['RT004M4'])
clean.iloc[:, 6:10].value_counts()

#clean Irrelevant Inputs
clean = clean.replace(-8, np.nan)
clean[clean == -8].count()

clean = clean.replace(8, np.nan)
clean[clean == 8].count()

clean = clean.replace(9, np.nan)
clean[clean == 9].count()

clean = clean.replace(98, np.nan)
clean[clean == 98].count()

clean = clean.replace(99, np.nan)
clean[clean == 99].count()

clean = clean.replace(9999998, np.nan)
clean[clean == 9999998].count()

clean = clean.replace(9999999, np.nan)
clean[clean == 9999999].count()

clean = clean.replace(999999998, np.nan)
clean[clean == 999999998].count()

clean = clean.replace(999999999, np.nan)
clean[clean == 999999999].count()

clean = clean.replace(999998, np.nan)
clean[clean == 999998].count()

clean = clean.replace(999999, np.nan)
clean[clean == 999999].count()

plt.style.use('seaborn')
clean.iloc[:, 4:14].plot.hist(subplots=True, legend=True, figsize=(12, 10))

#Dealing with Outliers - RT013 
clean[clean['RT013'] > 7000000].describe() #5 attributes have values
clean.loc[clean["RT013"] > 7000000, "RT013"] = np.nan
clean[clean['RT013'] > 7000000].describe() #cleaned

#Distribution of Cleaned RT013
filtered_RT013 = clean["RT013"][~np.isnan(clean["RT013"])]
fig, ax = plt.subplots(figsize=(20,6))
sns.boxplot(x=filtered_RT013)
plt.title('Cleaned Box Plot: total face value of life insurance policies', fontsize=15)
plt.xlabel('(millions of dollars)', fontsize=14)
plt.show()

histRT013 = clean["RT013"].plot(kind='hist', title='Distribution of Cleaned RT013')
clean["RT013"].skew() #still extremely skewed 

clean[clean['RT013'].between(1,999)].describe() #26 attributes have semantic anomalies in small value life insurance policiesvalues
clean.loc[clean["RT013"].between(1,999), "RT013"] = np.nan
clean[clean['RT013'].between(1,999)].describe() #cleaned

#Distribution of Second Cleaned RT013
filtered_RT013 = clean["RT013"][~np.isnan(clean["RT013"])]
fig, ax = plt.subplots(figsize=(20,6))
sns.boxplot(x=filtered_RT013)
plt.title('Cleaned Box Plot: total face value of life insurance policies', fontsize=15)
plt.xlabel('(millions of dollars)', fontsize=14)
plt.show()

histRT013 = clean["RT013"].plot(kind='hist', title='Distribution of Cleaned RT013')
clean["RT013"].skew() #still extremely skewed 

#Dealing with Outliers - RQ020 
clean[clean["RQ020"].between(1,14999)].describe() #839 attributes have potential semantic anomalies
MinWage = clean["RQ020"][clean["RQ020"] < 15000]
histRQ020 = MinWage.plot(kind='hist', title='Distribution of RQ020<15,000')

cleanHypo = clean
cleanHypo.loc[cleanHypo["RQ020"].between(1,14999), "RQ020"] = np.nan

#Distribution of Hypthetical Cleaned RQ020
filtered_RQ020 = cleanHypo["RQ020"][~np.isnan(cleanHypo["RQ020"])]
fig, ax = plt.subplots(figsize=(20,6))
sns.boxplot(x=filtered_RQ020)
plt.title('Hypthetical Cleaned Box Plot: wage and salary income in the Last Calendar Year', fontsize=15)
plt.xlabel('(millions of dollars)', fontsize=14)
plt.show()

cleanHypo["RQ020"].skew() #still extremely skewed

#Cleaning missing values
clean.dropna(how='all', subset = ['RT001', 'RT003', 'RT004M1', 'RT004M2','RT004M2','RT004M3','RT004M4','RT005', 'RT006', 'RT008', 'RT011', 'RT013', 'RQ020', 'RB017M', 'RB020', 'RB022', 'RB089M1M', 'RQ454', ], inplace = True)#remove rows of non-id attributes with null

subset = clean.iloc[:, 4:21]

subset.isna().sum()

msno.bar(subset)

#RT001
subsetNoWill = subset[subset["RT001"].isin([3, 5])]
subsetNoWill.isna().sum()

#RT004M1-M4
subsetNoKid = subset[(subset["RT004M1"] != 1) & (subset["RT004M1"] != 2) & (subset["RT004M1"] != 3) & (subset["RT001"] != 3) & (subset["RT001"] != 5)]
subsetNoKid.isna().sum()

subsetNoKidNoWill = subset[(subset["RT004M1"] == 1) | (subset["RT004M1"] == 2) | (subset["RT004M1"] == 3) | (subset["RT001"] == 1) | (subset["RT001"] == 2)]
subsetNoKidNoWill.isna().sum()

subsetNoKidNoWill = subsetNoKidNoWill.iloc[:, 0:9]
msno.bar(subsetNoKidNoWill)

#Will Relationships
subsetWill = subset[subset["RT001"].isin([1, 2])]
subsetWill = subsetWill.iloc[:, 0:9]

subsetWill.groupby("RT001")["RT001"].count()

subsetWill.groupby("RT003")["RT003"].count()

subsetWill.groupby("RT004M1")["RT004M1"].count()

subsetWill.groupby("RT004M2")["RT004M2"].count()

subsetWill.groupby("RT004M3")["RT004M3"].count()

subsetWill.groupby("RT005")["RT005"].count()

subsetWill.groupby("RT006")["RT006"].count()

subsetWill.groupby("RT008")["RT008"].count()

subsetWill.iloc[:,  2: 8]

# Logistic regression RT003
from sklearn import linear_model
#split the data into train and test sets with a ratio of 0.7:0.3. 
cleanWill = subsetWill.dropna()

X = cleanWill[:,  2: 8]
y = cleanWill.RT003 #RT003 is our dependent variable
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

lr = linear_model.LogisticRegression()
lr.fit(X_train, y_train)
lr.coef_ #ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 1.0


#life insurance
subsetInsurance = clean[["RT011", "RT013"]]
subsetInsurance = subsetInsurance[(subsetInsurance["RT011"] != 1)]
subsetInsurance.isna().sum()

subsetNoKidNoWill = subsetNoKidNoWill.iloc[:, 0:9]
msno.bar(subsetNoKidNoWill)


clean.to_csv('clean.csv', index=False)


# dropping duplicate values
data = pd.read_csv("compressed data.csv")
len(data)

data.drop_duplicates(keep="first",inplace=True)
 
# length after removing duplicates
len(data)
data.to_csv('compressed data2.csv', index=False)


#Removing 

