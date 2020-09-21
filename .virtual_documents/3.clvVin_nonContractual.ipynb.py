import pandas as pd
from lifetimes.datasets import load_transaction_data
from lifetimes.utils import summary_data_from_transaction_data

get_ipython().run_line_magic("matplotlib", " inline")


df = pd.read_excel('../dados/vinhos/vendas.xlsx')


# criar um id
df = df.reset_index()
df.head()


df.columns


df.columns = ['index', 'customerId', 'date', 'value','IVA', 'discountFinan', 'discountComer','valueWithIVA']


df.head()


#calcular a data maior
df.date.max()


df[df.customerId==7].head()


df[df.customerId==2].value.sum()


df.describe()


df.agg({'customerId':['count','mean','std','min','max']})


df.date.min()


df.date.max()


summary = summary_data_from_transaction_data(df,customer_id_col='customerId',datetime_col='date',freq='M',
                                             monetary_value_col='value',observation_period_end='2019-09-11')
summary.head()


summary.describe()


summary.shape


summary.columns


summary['monetary_value'].astype(int).head()


from lifetimes import BetaGeoFitter

bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])
print(bgf)


bgf.summary


from lifetimes.plotting import plot_frequency_recency_matrix
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']= [10,10]

plot_frequency_recency_matrix(bgf,title="");


from lifetimes.plotting import plot_probability_alive_matrix

plot_probability_alive_matrix(bgf,title="");


t=12
summary['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, summary['frequency'], summary['recency'], 
                                                        summary['T'])
summary.sort_values(by='predicted_purchases').tail(10)


from lifetimes.plotting import plot_period_transactions
plt.rcParams['figure.figsize']= [12,3]

plot_period_transactions(bgf);


from lifetimes.utils import calibration_and_holdout_data

summary_cal_holdout = calibration_and_holdout_data(df, 'customerId', 'date',calibration_period_end='2019-03-31',
                                                   observation_period_end='2019-09-12' )


df.date.min()


df.date.max()


summary_cal_holdout.head()


from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases

bgf.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])
plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout,title="");


summary.head(10)


summary.describe()


# Vamos prever o valor com o cliente número 11 está na posição 9
summary.iloc[9]


t = 12 #predict purchases in 10 periods
individual = summary.iloc[20]
# The below function is an alias to `bfg.conditional_expected_number_of_purchases_up_to_time`
bgf.predict(t, individual['frequency'], individual['recency'], individual['T'])
# 0.0576511


df.head()


from lifetimes.plotting import plot_history_alive

customerId = 3
days_since_birth = 24 #the number of time units since the birth we want to draw the p_alive
sp_trans = df.loc[df['customerId'] == customerId]
plot_history_alive(bgf, days_since_birth, sp_trans, 'date',freq='M');


df.columns


df.loc[df.customerId == 3].agg({'valueWithIVA':['sum','count','mean','std','min','max']})


summary.head(10)


summary_ggf = summary.loc[(summary.frequency > 0) & (summary.monetary_value > 0)]


summary_ggf.columns


summary_ggf[['frequency','monetary_value']].corr()


summary_ggf.monetary_value.hist()


from lifetimes import GammaGammaFitter

ggf = GammaGammaFitter(penalizer_coef=0.0)
ggf.fit(summary_ggf['frequency'],summary_ggf['monetary_value'])


ggf.conditional_expected_average_profit(summary_ggf['frequency'],summary_ggf['monetary_value']).head(10)


bgf.fit(summary_ggf['frequency'], summary_ggf['recency'], summary_ggf['T'])


bgf.fit(summary_ggf['frequency'], summary_ggf['recency'], summary_ggf['T'])

ggf.customer_lifetime_value(
    bgf, #the model to use to predict the number of future transactions
    summary_ggf['frequency'],
    summary_ggf['recency'],
    summary_ggf['T'],
    summary_ggf['monetary_value'],
    time=12, # months
    discount_rate=0.01 # monthly discount rate ~ 12.7% annually
).head(10)


ggf_CLV = ggf.customer_lifetime_value(
    bgf, #the model to use to predict the number of future transactions
    summary_ggf['frequency'],
    summary_ggf['recency'],
    summary_ggf['T'],
    summary_ggf['monetary_value'],
    time=12, # months
    discount_rate=0.01 # monthly discount rate ~ 12.7% annually
)


ggf_CLV.head(10)


# transformar Series em dataframe 
ggf_CLV=pd.DataFrame(ggf_CLV)
#ggf=pd.DataFrame(ggf)


dadosTodos=pd.merge(summary,ggf_CLV, left_index=True, right_index=True)


dadosTodos.shape


dadosTodos.head().reset_index()


X = dadosTodos.copy()


from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.spatial.distance import cdist
import numpy as np
#Finding optimal no. of clusters
clusters=range(1,20)
meanDistortions=[]
 
for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(X)
    prediction=model.predict(X)
    meanDistortions.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
 
#plt.cla()
plt.plot(clusters, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
#plt.title('Selecting k with the Elbow Method')


cluster = KMeans(n_clusters=3)

cluster.fit(X)
X['cluster']=cluster.predict(X)
print(X.cluster.value_counts())


X.columns


X.iloc[:,0].max()


from sklearn.decomposition import PCA

plt.rcParams['figure.figsize'] = [13, 4]

pca=PCA(n_components=2)
X['X_pca']=pca.fit_transform(X)[:,0] #todas as linhas da primeira coluna com redução
X['Y_pca']=pca.fit_transform(X)[:,1] #todas as linhas da segunda coluna com redução

fig, ax = plt.subplots()
for cluster in X.cluster.unique():
    x = X['X_pca'].loc[X.cluster == cluster]
    y = X['Y_pca'].loc[X.cluster == cluster]
    #scale = 200.0 * rand(n)
    ax.scatter(x, y, label=cluster, alpha=1, edgecolors='none')

ax.legend()
ax.grid(True)
ax.set_title('Clusters clientes')
plt.show()


X.columns


X.groupby(['cluster']).agg({'frequency':['count','mean','std'],
                            'recency':['mean','std'],
                            'T':['mean','std'],
                            'monetary_value':['mean','std'],
                            'predicted_purchases':['mean','std'],
                            'clv':['mean','std']})


X.columns


Y = X.copy()


cols = ['clv']


from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.spatial.distance import cdist
import numpy as np
#Finding optimal no. of clusters
clusters=range(1,20)
meanDistortions=[]
 
for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(Y[cols])
    prediction=model.predict(Y[cols])
    meanDistortions.append(sum(np.min(cdist(Y[cols], model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
 
#plt.cla()
plt.plot(clusters, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion');
#plt.title('Selecting k with the Elbow Method');


cluster = KMeans(n_clusters=3)

cluster.fit(Y[cols])
Y['cluster']=cluster.predict(Y[cols])
print(Y.cluster.value_counts())


Y.columns


Y.shape


Y.tail()


Y[cols[0]]


from sklearn.decomposition import PCA

plt.rcParams['figure.figsize'] = [13, 4]

fig, ax = plt.subplots()
for cluster in Y.cluster.unique():
    y = Y['monetary_value'].loc[Y.cluster == cluster]
    x = Y[cols[0]].loc[Y.cluster == cluster] #CLV
    #scale = 200.0 * rand(n)
    ax.scatter(x, y, label=cluster, alpha=1, edgecolors='none')

ax.legend()
ax.grid(True)
#ax.set_title('Clusters customer')
ax.set_xlabel('Customer Lifetime Value')
ax.set_ylabel('Monetary Value')

# Colocar a legenda eixo X e Y
plt.show()


Y.groupby(['cluster']).agg({'frequency':['count','mean','std'],
                            'recency':['mean','std'],
                            'T':['mean','std'],
                            'monetary_value':['mean','std'],
                            'predicted_purchases':['mean','std'],
                            'clv':['mean','std']})


Y.groupby(['cluster']).agg({'frequency':['count','mean','std'],
                            'predicted_purchases':['mean','std'],
                            'monetary_value':['mean','std'],
                            'clv':['mean','std']})


Y.head()


import datetime
df['year'] = df['date'].apply(lambda x: x.date().year)


df.head()


import seaborn as sns
sns.barplot(x='year',y='value',data=df,estimator=np.sum)


df.columns


df.iloc[0]


df.iloc[1:4]


df.iloc[1:4, 0:3]


x='teste.2222,222'


type(x.split('.')[1].split(',')[0])
