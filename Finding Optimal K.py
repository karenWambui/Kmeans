import pandas as pd
from sklearn.cluster import KMeans

#loading the data
data=pd.read_csv('OnlineRetail.csv',encoding='latin1')
print(data.head(5))
#print(data.to_string())
#shape of data columns and rows
#print(data.shape)

#print("Hello this is",data.info())
#print(data.columns)
#print(data.describe())
print(data.isnull().sum())
df_null=round(100*(data.isnull().sum())/len(data),2)
print(df_null)
data=data.dropna()
#data=data.dropna('StockCode')
print("This is dropped data",data)
#change the type 
data['CustomerID']=data['CustomerID'].astype(str)
#calculate all null
print(data.isnull().sum())
#Calculate Amount
data['Amount']=data['Quantity'] *data['UnitPrice']
#print(data.head())
#Group by CustomerId
dm =data.groupby('CustomerID')['Amount'].sum()
#grouping
da =data.groupby('Description')['Quantity'].sum()
print(da)
#da=da.reset_index()
print(da.head)
dc =data.groupby('Country')['Amount'].sum()
print(dc.idxmax())
#frequently sold product
dff=data.groupby('Description')['InvoiceNo'].count().sort_values(ascending=False)
print('Heyyy',dff)
#Convert to datetime to proper datatype
data['InvoiceDate'] =pd.to_datetime(data['InvoiceDate'],format='%m/%d/%Y %H:%M')
print("dates",data.head(5))
print(data.InvoiceDate)
#last transaction date
max_date= max(data['InvoiceDate'])
print(max_date)
min_date= min(data['InvoiceDate'])
print(min_date)
days = max_date - min_date
print(days)
tsales = max_date - pd.Timedelta(days=30)
print(tsales)
days1 = max_date-tsales
print(days1)

# Print the total sales for each month

total_sales=data[(data['InvoiceDate']>=tsales)&(data['InvoiceDate']<=max_date)]['Amount'].sum()
print("Total sales of the last month:",total_sales)

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from pandas.plotting import scatter_matrix

data2=data.groupby("StockCode").agg({"Quantity":"sum","UnitPrice":"sum"}).reset_index()
#Standardizing so that the comparison is fair
#importing standardscaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
data_3=scaler.fit_transform(data2[["Quantity","UnitPrice"]])
#ploting
data.plot(kind='box',subplots=True,sharex=False,sharey=False)
plt.show()
kmeans = KMeans(n_clusters=4, max_iter=50, random_state=42)
wcss = []  # Within-cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=50, random_state=42)
    kmeans.fit(data_3)
    wcss.append(kmeans.inertia_)

# Ploting the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# Based on the analysis
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, max_iter=50, random_state=42)
kmeans.fit(data_3)






