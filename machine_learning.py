#!/usr/bin/env python
# coding: utf-8

# In[1]:


# NLP Emotion


# In[2]:


import warnings


warnings.filterwarnings("ignore")


# # NLP (Natural language processing)

# In[3]:


#  NLP ----> forcasting on basic of Text data ( feedback )
# steps to be followed 

#read the file
#sp=pd.read_csv(r'C:\Users\Vijay\OneDrive\Documents\phytonpanda\spam1.csv',encoding='latin-1')
#sp.isnull().sum()[sp.isnull().sum()>0]
#sp=sp.loc[:,['v1','v2']]--------------> take usefull col.
#sp.head()
#sp=sp.rename(columns={'v1':'y','v2':'x'})------> rename as x and y
#sp.head()
#sp=sp.replace({'ham':0,'spam':1})-------------> y target varible as 0 and 1
#sp

#sp.x=sp.x.str.lower() ------------> all x varible in lower case


#import nltk   # NLP
#nltk.download('stopwords')
#from nltk.corpus import stopwords
#len(stopwords.words('english')) # only stopwords
# stopwords remove 

#import string
#string.punctuation# only punctuation
#l1=list(stopwords.words('english'))
# before buliding the model remove the stopwords and puntuation

#def text_process(mess):
#    """
#    2.remove stopwords
#    3.returnr the list of clean yextwords

#    """
#    nopunc=[char for char in mess if char not in string.punctuation]
#    nopunc="".join(nopunc)

#    return[word for word in nopunc.split() if word not in l1]

#sp['x'].apply(text_process)
#--------------------------------------------------------------------> remove stopewords and punctuation
# from sklearn.feature_extraction.text import CountVectorizer------->tdm count all usefull words

#import timeit
#start=timeit.default_timer()
#bow_transformer=CountVectorizer(analyzer=text_process).fit(sp['x'])
#stop=timeit.default_timer()
#execution_time=stop-start
#print('program execution in ',execution_time)

#bow_transformer.vocabulary_-------------------> all as text to number

#len(bow_transformer.vocabulary_)

#tdm=bow_transformer.transform(sp['x'])
#tdm.shape

#from sklearn.model_selection import train_test_split
#tdm_train,tdm_test,train_y,test_y=train_test_split(tdm,sp['y'],test_size=.2)

#from sklearn.naive_bayes import MultinomialNB
#nav=MultinomialNB()
#nav.fit(tdm_train,train_y)
#pred=nav.predict(tdm_test)
#from sklearn.metrics import confusion_matrix,accuracy_score
#tab=confusion_matrix(test_y,pred)
#tab
# accuracy_score(test_y,pred)
#------------------------------------------------------------------------->

# from sklearn.ensemble import RandomForestClassifier
# ran=RandomForestClassifier( n_estimators=20)
# ran.fit(tdm_train,train_y)
# pred_fc=ran.predict(tdm_test)
# tab=confusion_matrix(test_y,pred_fc)
# tab
# accuracy_score(test_y,pred_fc)
#------------------------------------------------------------------------->

# from sklearn.linear_model import LogisticRegression
# logreg=LogisticRegression()
# logreg.fit(tdm_train,train_y)
# pred_log=logreg.predict(tdm_test)
# tab=confusion_matrix(test_y,pred_log)
# tab
# accuracy_score(test_y,pred_log)



# To be taken as note
# 1) data should be Text variables as x and y should have text convert to number 
# 2) remove the stopwords remove the puncetion
# 3) count words TDM from start to stop it get converted into number
# 4) train test split 
# 5) do any bineary model


# In[4]:


import pandas as pd
import numpy as np


# # emotion_nlp

# In[5]:


pd.read_csv(r"C:\Users\user\Desktop\MLA\emotion_nlp.csv")


# In[6]:


em=pd.read_csv(r"C:\Users\user\Desktop\MLA\emotion_nlp.csv")


# In[7]:


em.Emotions.value_counts()


# In[8]:


em.isnull().sum()[em.isnull().sum()>0]


# In[9]:


em.Emotions.unique()


# In[10]:


em=em.rename(columns={'Emotions':'y','Text':'x'})


# In[11]:


em.head()


# In[12]:


em=em.replace({'sadness':1,'anger':2,'love':3,'surprise':4,'fear':5,'joy':6})
em


# In[13]:


em.x=em.x.str.lower() 
em.x


# In[14]:


import nltk   # NLP
nltk.download('stopwords')


# In[15]:


from nltk.corpus import stopwords
len(stopwords.words('english')) # only stopwords
# stopwords remove them


# In[16]:


import string
string.punctuation# only punctuation
len(string.punctuation)


# In[17]:


l1=list(stopwords.words('english'))
l1


# In[18]:


def text_process(mess):
    """
    1.remove punction
    2.remove stopwords
    3.return the list of clean yextwords

    """
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc="".join(nopunc)

    return[word for word in nopunc.split() if word not in l1]


# In[19]:


# nopunc = "".join(nopunc) joins the list of characters back into a single string without punctuation.


# In[20]:


#The list comprehension [word for word in nopunc.split() if word not in l1] splits
#the string nopunc into words and includes each word in the result only if it's not in l1.


# In[21]:


em['x'].apply(text_process)


# In[22]:


from sklearn.feature_extraction.text import CountVectorizer
#powerful tool for converting a collection of text documents into a matrix of token counts, facilitating the
#transformation of text data into numerical features suitable for machine learning models

#CountVectorizer performs tokenization and counts the frequency of each word in a document,
#effectively creating a "Bag of Words" (BoW) model.


# In[23]:


import timeit
start=timeit.default_timer()

bow_transformer=CountVectorizer(analyzer=text_process).fit(em['x'])

stop=timeit.default_timer()
execution_time=stop-start
print('program execution in ',execution_time)


# In[1]:


bow_transformer.vocabulary_


# In[ ]:


len(bow_transformer.vocabulary_)


# In[26]:


tdm=bow_transformer.transform(em['x'])

#In this case, tdm will be a sparse matrix representing the term frequencies of the
#words in new_docs based on the vocabulary learned from corpus.


# In[27]:


tdm.shape
# tdm is like our X varible


# In[28]:


from sklearn.model_selection import train_test_split
tdm_train,tdm_test,train_y,test_y=train_test_split(tdm,em['y'],test_size=.2)


# In[29]:


from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()


# In[30]:


nb.fit(tdm_train,train_y)


# In[31]:


pred_nb=nb.predict(tdm_test)


# In[32]:


from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,classification_report


# In[33]:


tab_nb=confusion_matrix(test_y,pred_nb)
tab_nb


# In[34]:


accuracy_score(test_y,pred_nb)*100


# In[35]:


print(classification_report(test_y,pred_nb))


# In[36]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()


# In[37]:


rfc.fit(tdm_train,train_y)


# In[38]:


pred_rfc = rfc.predict(tdm_test)


# In[39]:


tab_rfc = confusion_matrix(test_y,pred_rfc)
tab_rfc


# In[40]:


accuracy_score(test_y,pred_rfc)


# In[41]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(tdm_train , train_y)



# In[42]:


pred_log = logreg.predict(tdm_test)


# In[43]:


from sklearn.metrics import confusion_matrix


# In[44]:


tab_log = confusion_matrix(test_y , pred_log)
tab_log


# In[45]:


accuracy_score(test_y , pred_log)


# In[ ]:





# In[46]:


import matplotlib.pyplot as plt


# In[47]:


get_ipython().system('pip install WordCloud')


# In[48]:


from wordcloud import WordCloud
cloud=WordCloud(stopwords=stopwords.words('english'),max_words=30).generate(str(em['x']))
plt.figure(figsize=(10,10))
plt.imshow(cloud)


# In[49]:


em.y.value_counts()


# In[50]:


em_emo_df=em[em.y==1]


# In[51]:


from wordcloud import WordCloud
cloud=WordCloud(stopwords=stopwords.words('english'),max_words=40).generate(str(em_emo_df['x']))
plt.figure(figsize=(10,10))
plt.imshow(cloud)


# In[52]:


##-->em=em.replace({'sadness':1,'anger':2,'love':3,'surprise':4,'fear':5,'joy':6})


# In[53]:


em_emo_df1=em[em.y==2]


# In[54]:


cloud=WordCloud(stopwords=stopwords.words('english'),max_words=40).generate(str(em_emo_df1['x']))
plt.figure(figsize=(10,10))
plt.imshow(cloud)


# In[55]:


em_emo_df2=em[em.y==3]


# In[56]:


cloud=WordCloud(stopwords=stopwords.words('english'),max_words=40).generate(str(em_emo_df2['x']))
plt.figure(figsize=(10,10))
plt.imshow(cloud)


# In[57]:


##-->em=em.replace({'sadness':1,'anger':2,'love':3,'surprise':4,'fear':5,'joy':6})


# In[ ]:




# Data
1) spam1
2)Trip_advisor_review
3)fake_job_postings
4)UpdatedResumeDataSet
# # TextBlob

# In[58]:


import nltk


# In[59]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[60]:


nltk.download('vader_lexicon') #dictionary or words


# In[61]:


# vader is used for sentiment analysis --> +ve or -ve


# In[62]:


sent=SentimentIntensityAnalyzer()


# In[63]:


score=sent.polarity_scores('coffee was good and my brain cells finally got activated')
score


# In[64]:


#Proportion of text that is negative
#Proportion that is neutral
#Proportion that is positive
#Overall sentiment score, ranges from -1 (very negative) to +1 (very positive)


# In[65]:


sent.polarity_scores('AI a opportunity or threat?')


# In[66]:


type(score)


# In[67]:


score.keys()


# In[68]:


score['compound']


# In[69]:


#trip=pd.read_csv(r"C:\Users\Vijay\OneDrive\Desktop\Files\Trip_advisor_review.csv")
tr=trip=pd.read_csv(r"C:\Users\user\Desktop\MLA\Trip_advisor_review.csv")
tr


# In[70]:


# consider you dont have rating column with you


# In[71]:


l1=list(tr['Review'])


# In[72]:


l1


# In[73]:


l2=[]
for i in l1:
    score=sent.polarity_scores(i)
    l2.append(score['compound'])


# In[74]:


len(l2)


# In[75]:


tr['score']=l2


# In[76]:


tr1=tr


# In[77]:


tr1 = tr1.drop('Rating', axis=1)


# In[78]:


20491-18862


# In[79]:


tr1['score'].describe()


# In[80]:


tr1[tr1['score']>0].shape

#20000 → number of rows where score > 0
#5 → number of columns in the DataFrame


# In[81]:


# Your dataset has mostly strongly positive sentiment, with some negative outliers.
# If you're analyzing customer reviews, for instance, this would suggest customers are generally very happy.
# Would you like a visualization (e.g., histogram or boxplot) of this distribution?


# # TextBlob

# In[82]:


get_ipython().system('pip install TextBlob')


# In[4]:


from textblob import TextBlob
score=TextBlob('food was not tasty and was bad')


# In[5]:


score.sentiment


# In[3]:


score=TextBlob('bad good')
score.sentiment


# In[86]:


score=TextBlob('')
score.sentiment
l1=list(tr['Review'])


# In[87]:


l2=[]
for i in l1:
    score=TextBlob(i)
    l2.append(score.sentiment[0])

# 0 polarity    


# In[88]:


l2


# In[89]:


tr['score'].describe()


# In[90]:


score.sentiment


# In[91]:


len(tr[tr['score']>0])
# Count of positive reviews


# In[92]:


len(tr[tr['score']<0])
# Count of negative reviews


# In[93]:


tr.shape


# In[94]:


18862/20491*100
#This is calculating the percentage of something, most likely positive reviews:


# In[ ]:





# In[6]:


get_ipython().system('pip install  spacy')


# In[8]:


lang=spacy.load('en_core_web_sm')


# In[9]:


import spacy

spacy.cli.download("en_core_web_sm")

#Loads the English small model (en_core_web_sm), 
#which includes tokenization, part-of-speech tagging, named entity recognition, etc.


# In[10]:


doc=lang('stock market has been moving upwaards from last sessions')


# In[ ]:


type(doc)


# In[ ]:


for i in doc:
    print(i.text)


# In[ ]:


# pos part of the speach
#This prints each token along with its part of speech, like NOUN, VERB, ADJ, etc.


# In[ ]:


for i in doc:
    print(i.text,i.pos_)


# In[ ]:


doc3 = lang("my website is www.abc.com and my email id is xyz@gmail.com this is just an example")

for token in doc3:
    print(token)


# In[ ]:





# In[ ]:


doc2=lang('stock market has been moving upword from last few session . i hope it continues the upword journey')


# In[ ]:


for j in doc2.sents:
    print(j)


# In[ ]:


doc6=lang('Vijay live in Pune which is part of Maharashtra and which is part of India')


# In[ ]:


for token in doc6.ents:
    print(token)

#This prints named entities in the sentence.    


# In[ ]:


for token in doc6.ents:
    print(token)
    print(token.label_)
    print(str(spacy.explain(token.label_)))
    print('----------------------')


# In[ ]:


doc6=lang('Kangaroo Won the World Cup')
for token in doc6.ents:
    print(token)




# In[ ]:


for token in doc6.ents:
    print(token)
    print(token.label_)
    print(str(spacy.explain(token.label_)))
    print('----------------------')


# In[ ]:





# In[11]:


import nltk
from nltk.stem.snowball import SnowballStemmer


# In[12]:


stemmer=SnowballStemmer(language='english')


# In[ ]:


word_list={'swim','swimmer','matches','mens','langhing','loving','humens','indians','finishing','watchs','fans','writting'}


# In[ ]:


for word in word_list:
    print(word,'its stemming is-->','\t',stemmer.stem(word))


# In[ ]:





# In[ ]:





# # Time Series

# In[ ]:


#  Time Series----> forcasting for upcomig date
# steps to be followed --> unsupervised model

#data import 
#data cleaning-->null,replace

#airpas=pd.read_csv(r"C:\Users\Vijay\OneDrive\Documents\phytonpanda\AirPassengers.csv")
#airpas.isnull().sum()[airpas.isnull().sum()>0]
#airpas.info()
#airpas.Month=pd.to_datetime(airpas.Month)
#airpas=airpas.set_index('Month')
#airpas
#---------------------------------------------------------->
#plt.figure(figsize=(10,10))
#plt.plot(airpas.Passengers,marker='*')
#plt.grid()-------------->check stationarity

#plt.plot(airpas.Passengers.diff().diff().diff().diff().diff().diff().diff())------->done stationarity

#airpas_log=np.log(airpas)--> converted to log

#plt.plot(airpas_log.Passengers.diff())-------> log then easy to convert in stationarity

#from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
#plot_acf(airpas_log.Passengers)
#plot_pacf(airpas_log.Passengers)---------------> need to check how P,d,q comes

#import pmdarima 
#from pmdarima import auto_arima

#auto_arima(airpas_log)
#from statsmodels.tsa.statespace.sarimax import SARIMAX

#model_sarima=SARIMAX(airpas_log,order=(4, 1, 3))---------->4=p(AR),1=d(I),3=q(MA)
#result=model_sarima.fit()
#pred_log=result.predict(start=144,end=167)---------------> satart and end upcoming two years forcasting 
#pred=np.exp(pred_log)
#pred=np.round(pred)
#pred

#plt.figure(figsize=(10,10))
#plt.plot(airpas,color='red',label='49 -60')
#plt.plot(pred,color='green',label='61 -62')
#plt.grid()
#plt.legend()
#-----------------------------------------> check plot plot should be same as first one if not go to seasonal_order

#auto_arima(airpas_log,seasonal=True,m=12)
#model_sarima=SARIMAX(airpas_log,order=(2, 0, 0),seasonal_order=(0, 1, 1, 12),)----->12= months
#result=model_sarima.fit()
#pred_log=result.predict(start=144,end=167)---> upcoming months forcasting
#pred=np.exp(pred_log)
#pred=np.round(pred)
#pred

#plt.figure(figsize=(10,10))
#plt.plot(airpas,color='red',label='49 -60')
#plt.plot(pred,color='green',label='61 -62')
#plt.grid()
#plt.legend()
#-----------------------------------------> check plot plot should be same as first one

#pd.read_csv(r"C:\Users\Vijay\OneDrive\Documents\phytonpanda\AirPassengers.csv")
#airpas=pd.read_csv(r"C:\Users\Vijay\OneDrive\Documents\phytonpanda\AirPassengers.csv")
#airpas.Month=pd.to_datetime(airpas.Month)
#airpas=airpas.set_index('Month')
#airpas

#airpas_log=np.log(airpas)
# split train and test manul
#airpas_train=airpas_log.iloc[0:132]
#airpas_test=airpas_log.iloc[132:144]

#auto_arima(airpas_train, m=12,seasonal=True,)--------------------------->12= months
#model_sarima=SARIMAX(airpas_train,order=(2, 0, 0),seasonal_order=(0, 1, 1, 12),)
#result=model_sarima.fit()
#pred_log=result.predict(start=132,end=143,)-----> upcoming months
#pred=np.exp(pred_log)
#pred=np.round(pred)

#actual=np.exp(airpas_test.Passengers)

#err=actual-pred

#from sklearn.metrics import mean_absolute_percentage_error
#mape=mean_absolute_percentage_error(actual,pred)*100------------------------> find mape and acc
#acc=100-mape
#acc

#plt.plot(actual,color='red',label='actual',marker='*')
#plt.plot(pred, color='green',label='predicted',marker='*')
#plt.legend()


# To be taken as note
# 1) data should be numerical variables only and should have date 
# 2) first take plot and check stationarity if not then do in log (diff())
# 3) then plot plot_acf,plot_pacf
# 4) then do auto_arima without seasonal check plot
# 5) then do SARIMA with seasonal check plot
# 6) then do train test split and find mape,acc

# models 
# 1) AR-->p
# 2) MA-->q
# 3) ARIMA--->i-->differencing levels
# 4) SARIMA--->seasonal
# 5) SARIMAX---->exog
#-------------> after every model do plot and check wheather the graph is going in same way or not


# # AirPassengers

# In[24]:


import pandas as pd


# In[25]:


pd.read_csv(r"C:\Users\user\Desktop\MLA\AirPassengers.csv")


# In[26]:


airpas=pd.read_csv(r"C:\Users\user\Desktop\MLA\AirPassengers.csv")


# In[27]:


# aim is to do forecast for next 24 months ( next 2 years 1961 and 1962)


# In[28]:


# 1> autoraima ---> value u get
# 2> SARIMAX ---->


# In[29]:


airpas.info()


# In[30]:


airpas.Month=pd.to_datetime(airpas.Month)
airpas.Month


# In[31]:


airpas=airpas.set_index('Month')


# In[32]:


airpas


# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[34]:


plt.figure(figsize=(10,6))
plt.plot(airpas.Passengers,marker='.')
plt.grid()
# uptrend, with in a year  values rise and fall
# but year on year values are increasing
# data is not statinary--> differcing


# In[ ]:





# In[13]:


plt.plot(airpas.Passengers.diff().diff().diff().diff().diff().diff().diff())


# In[35]:


import numpy as np


# In[36]:


airpas_log=np.log(airpas)


# In[37]:


plt.plot(airpas_log.Passengers.diff())
# after doing  1 level of diffrincing data is stationary
# now our base data is log data and when forecasting is done that would also be in log
# so to get the value in orinigal scale we need to take anti log


# In[38]:


# taking a log will give you a smoothing effect 


# In[ ]:





# In[39]:


# not mandatrory


# In[40]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[41]:


plot_acf(airpas_log.Passengers)


# In[21]:


plot_pacf(airpas_log.Passengers) # q=2,p=0,d=1


# In[ ]:





# In[42]:


from statsmodels.tsa.arima_model import ARIMA


# In[43]:


# let us find autoarima


# In[44]:


get_ipython().system('pip install pmdarima')


# In[45]:


import pmdarima 
from pmdarima import auto_arima


# In[26]:


auto_arima(airpas_log) # it will run model many times and give the values of p,d,q


# In[27]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[28]:


model_sarima=SARIMAX(airpas_log,order=(4,1,3))
model_sarima


# In[29]:


results= model_sarima.fit()
results


# In[31]:


pred_log=results.predict(start=144, end=167)
pred_log


# In[32]:


pred=np.exp(pred_log)
pred


# In[33]:


pred= np.round(pred)


# In[34]:


pred


# In[35]:


plt.figure(figsize=(10,6))
plt.plot(airpas , color = "red" , label = "49-60")
plt.plot(pred , color = "green" , label = "61-62")
plt.grid()
plt.legend()
# from this plot model is not doing good


# In[48]:


# arima can not handal seasonlity sarimax


# In[49]:


# lets bulid the modle where seasonality is considered


# In[50]:


auto_arima(airpas_log,seasonal=True,m=12)


# In[51]:


model_sarima=SARIMAX(airpas_log,order=(2,0,0),seasonal_order=(0,1,1,12))


# In[52]:


results=model_sarima.fit()


# In[ ]:


pred_log=results.predict(start=144,end=167)


# In[ ]:


pred=np.exp(pred_log)
pred=np.round(pred)
pred


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(airpas , color = "red" , label = "49-60")
plt.plot(pred , color = "green" , label = "61-62")
plt.grid()
plt.legend()


# In[ ]:


# we have build the model on entire data
# but now lets do the sampling and divided the data in train and test


# In[55]:


pd.read_csv(r"C:\Users\user\Desktop\MLA\AirPassengers.csv")


# In[59]:


airpas=pd.read_csv(r"C:\Users\user\Desktop\MLA\AirPassengers.csv")
airpas


# In[60]:


airpas.Month=pd.to_datetime(airpas.Month)


# In[61]:


airpas=airpas.set_index('Month')


# In[62]:


airpas_log=np.log(airpas)
airpas_log


# In[ ]:


# no radom sampling 
# sequancial sampling
# train 1949-1959
# test 1960


# In[63]:


airpas_train=airpas_log.iloc[0:132]
airpas_test=airpas_log.iloc[132:144]


# In[64]:


airpas_test


# In[65]:


# MAPE


# In[74]:


auto_arima(airpas_train,seasonal=True,m=12)


# In[75]:


model_sarima=SARIMAX(airpas_log,order=(2,0,0),seasonal_order=(0,1,1,12))


# In[77]:


results=model_sarima.fit()


# In[78]:


pred_log=results.predict(start=132,end=143)


# In[79]:


pred=np.exp(pred_log)


# In[80]:


pred=np.round(pred)
pred


# In[81]:


# err--> act-pred


# In[82]:


actual =np.exp(airpas_test.Passengers)
actual


# In[83]:


err=pred-actual


# In[84]:


err


# In[85]:


mape=np.mean(np.abs(err*100/ actual))


# In[86]:


mape


# In[87]:


acc=100-mape


# In[88]:


acc


# In[89]:


from sklearn.metrics import mean_absolute_percentage_error


# In[90]:


MAPE=mean_absolute_percentage_error(actual,pred)*100
ACC=100-MAPE
ACC


# In[91]:


plt.plot(actual,color='red',label='actual',marker='*')
plt.plot(pred, color='green',label='predicted',marker='*')
plt.legend()


# In[92]:


from statsmodels.tsa.seasonal import seasonal_decompose

dec=seasonal_decompose(airpas.Passengers)

dec.plot();


# In[ ]:





# # Data set
1.RestaurantVisitors
2.Food_Products_Value
3.Alcohol_Sales
4.monthly_milk_production
# In[ ]:





# In[ ]:





# # Neural Network

# # trainRF

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


pd.read_csv(r"C:\Users\user\Desktop\MLA\trainRF.csv")


# In[ ]:


mp=pd.read_csv(r"C:\Users\user\Desktop\MLA\trainRF.csv")


# In[ ]:


mp.shape


# In[ ]:


mp.isnull().sum()[mp.isnull().sum()>0]


# In[ ]:


mp.price_range.value_counts()


# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


mp_train , mp_test = train_test_split(mp, test_size=.2)


# In[ ]:


mp_train_x = mp_train.drop(['price_range'], axis= 1)
mp_train_y = mp_train.price_range

mp_test_x = mp_test.drop(['price_range'], axis= 1)
mp_test_y = mp_test.price_range


# In[ ]:


get_ipython().system('pip install tensorflow')



# In[ ]:


import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Dropout


# In[ ]:


#Starts a sequential model where each layer is stacked one after the other.
#Adds Dense layers (fully connected layers) with 128 neurons each using ReLU activation.
#These layers learn patterns in the data.

#Output layer with 4 neurons (for 4 classes of price_range), using softmax to output class probabilities.

#Compiles the model using the Adam optimizer and sparse categorical crossentropy 
#loss (suitable for integer class labels).

#Dropout is a technique to prevent overfitting. You commented it out, but you could apply it between layers.


# In[ ]:


model =tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu)) # I/p layer 1
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu)) # I/p layer 2
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu)) # I/p layer 3
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))


model.add(tf.keras.layers.Dense(4,activation=tf.nn.softmax))# o/p layer

#model.add(Dropout(0.1))  # after o/p layer--> only you looks that model is over fitting


adam=tf.keras.optimizers.Adam(learning_rate=.001)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=["accuracy"])


# In[ ]:


model.fit(mp_train_x,mp_train_y ,epochs=40,validation_split=.2, batch_size=40)


# In[ ]:


# After every batch we are changing weight


# In[ ]:


pred=model.predict(mp_test_x)
pred


# In[ ]:


pred_classes = pred.argmax(axis=1)
pred_classes
#Picks the class with the highest probability (argmax).


# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[ ]:


tab=confusion_matrix(mp_test_y,pred_classes)
tab


# In[ ]:


accuracy_score(mp_test_y,pred_classes)


# In[ ]:


history=model.fit(mp_train_x,mp_train_y ,epochs=40,validation_split=.2, batch_size=40)
df1=pd.DataFrame(model.history.history)
df1


# In[ ]:


plt.figure(figsize=(10,10))
plt.plot(df1.loss,color='red',marker='*',label='train')
plt.plot(df1.val_loss,color='green',marker='.',label='validation')
plt.legend()


# In[ ]:


#Plots training loss and validation loss to check for overfitting or underfitting.

#Both training and validation loss decrease steadily during training.
#The gap between them stays small.
#Validation loss might slightly increase or flatten at the end.


# # Property_Price_Train    Linear regression

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


pd.read_csv(r"C:\Users\user\Desktop\MLA\Property_Price_Train.csv")


# In[ ]:


pt=pd.read_csv(r"C:\Users\user\Desktop\MLA\Property_Price_Train.csv")


# In[ ]:


pt.isnull().sum()[pt.isnull().sum()>0]


# In[ ]:





# In[ ]:


pt.Brick_Veneer_Type.fillna( 'None', inplace = True )
pt.Basement_Height.fillna( 'TA', inplace = True )
pt.Basement_Condition.fillna( 'TA', inplace = True )
pt.Exposure_Level.fillna( 'No', inplace = True )
pt.BsmtFinType1.fillna( 'Unf', inplace = True )
pt.BsmtFinType2.fillna( 'Unf', inplace = True ) 
pt.Electrical_System.fillna( 'SBrkr', inplace = True )
pt.Fireplace_Quality.fillna( 'Gd', inplace = True )
pt.Garage.fillna( 'Attchd', inplace = True )
pt.Garage_Built_Year.fillna( 2005, inplace = True ) ##
pt.Garage_Finish_Year.fillna( 'Unf', inplace = True )
pt.Garage_Quality.fillna( 'TA', inplace = True )
pt.Garage_Condition.fillna( 'TA', inplace = True )
pt.Pool_Quality.fillna( 'Gd', inplace = True )
pt.Fence_Quality.fillna( 'MnPrv', inplace = True )
pt.Miscellaneous_Feature.fillna( 'Shed', inplace = True )
pt.Lot_Extent.fillna( pt.Lot_Extent.mean(), inplace = True)
pt.Lane_Type.fillna( 'Grvl', inplace = True )
pt.Brick_Veneer_Area.fillna( pt.Brick_Veneer_Area.median(), inplace = True )



# In[ ]:


pt=pt.drop(['Id','Fireplace_Quality','Pool_Quality','Fence_Quality','Miscellaneous_Feature','Lane_Type'],axis=1)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()



# In[ ]:


pt[pt.select_dtypes(include='object').columns] = pt[pt.select_dtypes(include='object').columns].apply(le.fit_transform)



# In[ ]:


pt.shape


# In[ ]:


from sklearn.model_selection import train_test_split
pt_train, pt_test = train_test_split(pt, test_size= .2)



# In[ ]:


pt_train_x = pt_train.iloc[::,0:-1]
pt_train_y = pt_train.iloc[::,-1]

pt_test_x = pt_test.iloc[::,0:-1]
pt_test_y = pt_test.iloc[::,-1]


# In[ ]:


pt_test_x=np.array(pt_test_x)
pt_train_y=np.array(pt_train_y)
pt_train_x=np.array(pt_train_x)


# In[ ]:


#NumPy arrays are the standard format for numerical computation in Python.

#While Keras sometimes accepts pandas DataFrames, NumPy arrays are more 
#consistent and reliable, especially when working with lower-level TensorFlow operations.


# In[ ]:


pt_test_x.shape


# In[ ]:


import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Dropout


# In[ ]:


def  build_model():
    model =tf.keras.Sequential([
        tf.keras.layers.Dense(64,activation=tf.nn.relu , input_dim=74),
        tf.keras.layers.Dense(64,activation=tf.nn.relu ),
        tf.keras.layers.Dense(64,activation=tf.nn.relu ),
        tf.keras.layers.Dense(64,activation=tf.nn.relu ),
        tf.keras.layers.Dense(1)    

        ])
    optimizer = tf.keras.optimizers.RMSprop(.005)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae','mse'])
    return model    



# In[ ]:


#Takes 74 input features

#Passes them through 4 hidden layers (each with 64 neurons, ReLU activation)

#Outputs 1 continuous value

#Is trained using the RMSprop optimizer to minimize MSE


# In[ ]:


#This defines a sequential model, meaning the layers are stacked one after another in a straight line.
#64 neurons
#ReLU activation (tf.nn.relu)
#input_dim=74: your input data has 74 features per sample

#These are three more hidden layers, each with 64 neurons and ReLU activation
#Stacking multiple layers allows the network to learn more complex patterns

#A single neuron with no activation
#This is typical for a regression task,
#where you want a real-number output (e.g., predicting house price, temperature, etc.)

#RMSprop is an optimization algorithm (like Adam or SGD)
#Learning rate is set to 0.005 (controls how fast the model learns)

#loss='mse': The model is trained to minimize Mean Squared Error (MSE) — common for regression problems.
#metrics=['mae', 'mse']: You’ll track Mean Absolute Error and MSE during training and validation.


# In[ ]:





# In[ ]:


model = build_model()


# In[ ]:


model.fit(pt_train_x,pt_train_y,epochs=500)


# In[ ]:


pred_test=model.predict(pt_test_x)


# In[ ]:


#pred_train=model.predict(pt_train_x)


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error


# In[ ]:


r2_test = r2_score(pt_test_y, pred_test)
r2_test


# In[ ]:


mse_test = mean_squared_error(pt_test_y, pred_test)
mse_test


# In[ ]:


mape_test = mean_absolute_percentage_error(pt_test_y, pred_test)
mape_test*100


# In[ ]:


Acc=100-13
Acc


# In[ ]:


model.weights


# In[ ]:





# # Dataset-->
# ### CreditRisk59

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


pd.read_csv(r"C:\Users\user\Desktop\MLA\CreditRisk59.csv")


# In[ ]:


cd=pd.read_csv(r"C:\Users\user\Desktop\MLA\CreditRisk59.csv")


# In[ ]:


cd


# In[ ]:


cd.shape


# In[ ]:


cd.isnull().sum()[cd.isnull().sum()>0]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


cd_train , cd_test = train_test_split(cd, test_size=.2)


# In[ ]:


import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Dropout


# In[ ]:


model =tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu)) # I/p layer 1
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu)) # I/p layer 2
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu)) # I/p layer 3
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))


model.add(tf.keras.layers.Dense(4,activation=tf.nn.softmax))# o/p layer

#model.add(Dropout(0.1))  # after o/p layer--> only you looks that model is over fitting


adam=tf.keras.optimizers.Adam(learning_rate=.001)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=["accuracy"])


# # CNN

# In[ ]:





# In[ ]:





# In[ ]:


#  NN's ----> we can say next step of ML
# steps to be followed 

#read the file
#pd.read_csv(r'C:\Users\Vijay\OneDrive\Documents\phytonpanda\MNIST_Train.csv')
#mnist=pd.read_csv(r'C:\Users\Vijay\OneDrive\Documents\phytonpanda\MNIST_Train.csv')
#mnist.isnull().sum()[mnist.isnull().sum()>0]
#mnist1=mnist.iloc[::,1::]
#mnist1.head()
# image is just a number------------------->


#mnist1=np.array(mnist1)
#type(mnist1)
#mnist1.shape
#row=mnist1[445]
#len(row)
#plt.imshow(row.reshape(28,28))--->28*28 =784--------> show image at row 445

#for i in range(50):
#    plt.subplot(5,10,i+1)
#    plt.imshow(mnist1[i,:].reshape(28,28))
#    plt.axis('off')
#------------------------------------------------> it will show images from 0 to 49



# vj=cv2.imread(r'C:\Users\Vijay\Downloads\Tiger.JPEG ')-----------> read image from laptop
# vj.shape
# plt.imshow(vj)
# plt.axis('off')
#-----------------------------------------------------------NN's------------------------------------------------------>
# Neural Network
# MNIST_Train
# mnist=pd.read_csv(r'C:\Users\Vijay\OneDrive\Documents\phytonpanda\MNIST_Train.csv')
#train test split-------------->
# from sklearn.model_selection import train_test_split
# mnist_train,mnist_test=train_test_split(mnist,test_size=.2)

# mnist_train_x=mnist_train.iloc[:,1::]
# mnist_train_y=mnist_train.iloc[:,0]

# mnist_test_x=mnist_test.iloc[:,1::]
# mnist_test_y=mnist_test.iloc[:,0]
# converted into array
# mnist_train_x=np.array(mnist_train_x)
# mnist_train_y=np.array(mnist_train_y)
# mnist_test_x=np.array(mnist_test_x)


# model = tf.keras.models.Sequential() ------------->i/p
# model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))  ----> 1st hidden layer--->128 neurons-->relu
# model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))  ----> 2nd hidden layer
# model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))  ----> 3rd hidden layer

# model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))  --->o/p---->10=y varible

# #model.add(Dropout(0.2)) # after o/p layer--> only you looks that model is over fitting--> 20% neuron is removed after 
#every ittration 

# adam=tf.keras.optimizers.Adam(learning_rate=.001) # to do not miss optiment point
# model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# model.fit(mnist_train_x , mnist_train_y, epochs = 50, validation_split = .2, batch_size = 100)

# After every batch we are changing weight

# pred=model.predict(mnist_test_x)
# pred
# pred_classes=pred.argmax(axis=1)------------> each row predection value
# pred_classes

# from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,f1_score
# tab=confusion_matrix(mnist_test_y,pred_classes)
# tab
# accuracy_score(mnist_test_y,pred_classes)



# df1=pd.DataFrame(model.history.history)
# df1

# plt.figure(figsize=(10,10))
# plt.plot(df1.loss,color='red',marker='*',label='train')
# plt.plot(df1.val_loss,color='green',marker='.',label='validation')
# plt.legend()
# when plot goes up then it is over fitting

# To be taken as note
# 1) data should be any bineary or contenives or image
# 2) Input layer--> takes all x varibles
# 3) Hidden Layer-->added neuron and relu activation function
# 4) Output Layer-->add y varible and softmax
# 5) After every batch we are changing weight--> first random then calculated
# 6) hyperparameter-->epochs = 50, validation_split = .2, batch_size = 100
# 7) data and accuracy should be more 
# 8) dropout is use only when model is geating overfiting --->when plot goes up then it is over fitting


# In[2]:


import pandas as pd


# In[3]:


pd.read_csv(r"C:\Users\user\Desktop\MLA\MNIST_Train.csv")


# In[4]:


mnist=pd.read_csv(r"C:\Users\user\Desktop\MLA\MNIST_Train.csv")


# In[5]:


mnist.label.value_counts()


# In[6]:


mnist.isnull().sum()[mnist.isnull().sum()>0]


# In[7]:


mnist1=mnist.iloc[::,1::]


# In[8]:


mnist1.head()


# In[9]:


# image is just a number
##That image has 784 features (pixels) — because it's a flattened 28×28 image.


# In[10]:


import numpy as np


# In[11]:


mnist1=np.array(mnist1)


# In[12]:


type(mnist1)


# In[13]:


mnist1.shape


# In[14]:


row=mnist1[234]
#row
#You are selecting the 1000th image (index 999) from the MNIST training dataset.
#28*28


# In[15]:


len(row)
#This tells us each image has 784 pixels — because MNIST images are 28×28 pixels:

#You are looking at a single image.
#That image has 784 features (pixels) — because it's a flattened 28×28 image.
#The shape (784,) means it's a 1D array with 784 elements.

#28 pixels wide
#28 pixels tall


# In[ ]:





# In[16]:


import matplotlib.pyplot as plt


# In[ ]:





# In[17]:


plt.imshow(row.reshape(28,28))

#row is expected to be a 1D array with 784 elements (28×28).
#.reshape(28,28) converts the flat array into a 2D image format.


# In[18]:


for i in range(50):
    plt.subplot(5,10,i+1)
    plt.imshow(mnist1[i,:].reshape(28,28))
    plt.axis('off')

#row is expected to be a 1D array with 784 elements (28×28).
#.reshape(28,28) converts the flat array into a 2D image format.   
#Purpose: Start a loop to display the first 50 images from the dataset mnist1.
#Purpose: Set up a subplot grid of 5 rows and 10 columns.


# In[ ]:





# In[22]:


get_ipython().system('pip install opencv-python')


# In[23]:


import cv2


# In[25]:


vj=cv2.imread(r"C:\Users\user\Desktop\MLA\images\India.jpg")


# In[26]:


vj.shape


# In[27]:


plt.imshow(vj)
plt.axis('off')


# In[ ]:





# In[ ]:





# # MNIST_Train

# In[29]:


get_ipython().system('pip install opencv-python')


# In[ ]:





# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from tensorflow.keras.utils import to_categorical
import os


# In[31]:


pd.read_csv(r"C:\Users\user\Desktop\MLA\MNIST_Train.csv")


# In[32]:


mnist=pd.read_csv(r"C:\Users\user\Desktop\MLA\MNIST_Train.csv")


# In[33]:


from sklearn.model_selection import train_test_split
mnist_train,mnist_test=train_test_split(mnist,test_size=.2)


# In[34]:


mnist_train_x=mnist_train.iloc[:,1::]
mnist_train_y=mnist_train.iloc[:,0]

mnist_test_x=mnist_test.iloc[:,1::]
mnist_test_y=mnist_test.iloc[:,0]


# In[35]:


mnist_train_x=np.array(mnist_train_x)
mnist_train_y=np.array(mnist_train_y)

mnist_test_x=np.array(mnist_test_x)

#Required for feeding into TensorFlow.

mnist_train_x=mnist_train_x.reshape(-1,28,28,1)
#Reshapes flat vectors into 28×28×1 (height, width, channels) for CNN input.



# In[36]:


mnist_x=tf.keras.utils.normalize(mnist_train_x)
#Normalizes pixel values to range [0, 1] (divides by 255).


mnist_train_y=to_categorical(mnist_train_y)
#One-Hot Encode Labels
#Converts labels like 3 into [0, 0, 0, 1, 0, ..., 0] for categorical crossentropy.
#1. Match the model's output shape
#Neural networks for classification typically end in a softmax layer with 10 units (one per digit 0–9).
#This softmax layer outputs a probability distribution over the classes (shape (batch_size, 10)).
mnist_train_y


# In[37]:


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())  
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))


#Conv2D layer: 16 filters of 5×5 size → extracts image features.
#MaxPool2D: down-samples (reduces image size to prevent overfitting).
#Flatten: converts image to 1D vector.
#Dense (128): fully connected hidden layer.
#Dropout (0.2): randomly disables 20% of neurons during training (to prevent overfitting).
#Output layer (10 neurons): one for each digit (0–9), softmax for probability output.


# In[38]:


adam=tf.keras.optimizers.Adam(learning_rate=0.001)


# In[39]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


#Adam optimizer with learning rate 0.001.
#Loss: categorical_crossentropy (used for one-hot encoded labels).
#Metric: accuracy.


# In[40]:


model.fit(mnist_train_x,mnist_train_y,validation_split=0.2,batch_size=64,epochs=5)

#Trains for 5 epochs.
#Uses 64 samples per batch.
#Reserves 20% of training data for validation.


# In[41]:


mnist_test_x = mnist_test_x.reshape(-1, 28, 28, 1)
#mnist_test_x
#Reshapes the test data like training data (for CNN input).
##Reshapes flat vectors into 28×28×1 (height, width, channels) for CNN input.
#mnist_test_x = tf.keras.utils.normalize(mnist_test_x)
mnist_test_x


# In[42]:


pred=model.predict(mnist_test_x)
pred

#Gets predicted probabilities and converts to class labels (0–9).


# In[43]:


pred_classes=pred.argmax(axis=1)
pred_classes


# In[44]:


from sklearn.metrics import confusion_matrix,accuracy_score
a=confusion_matrix(mnist_test_y,pred_classes)
a


# In[45]:


accuracy_score(mnist_test_y,pred_classes)


# In[46]:


from sklearn.metrics import classification_report


# In[47]:


print(classification_report(mnist_test_y,pred_classes))


# In[ ]:





# In[48]:


get_ipython().system('pip install tensorflow')


# In[49]:


import pandas as pd


# In[50]:


mnist=pd.read_csv(r"C:\Users\user\Desktop\MLA\MNIST_Train.csv")
mnist


# In[51]:


import tensorflow as tf
import keras


# In[52]:


from sklearn.model_selection import train_test_split
mnist_train,mnist_test=train_test_split(mnist,test_size=.2)


# In[53]:


mnist_train_x=mnist_train.iloc[:,1::]
mnist_train_y=mnist_train.iloc[:,0]

# pixel data
# digit labels


# In[54]:


mnist_test_x=mnist_test.iloc[:,1::]
mnist_test_y=mnist_test.iloc[:,0]


# In[55]:


mnist_train_x=np.array(mnist_train_x)
mnist_train_y=np.array(mnist_train_y)

mnist_test_x=np.array(mnist_test_x)


# In[56]:


from keras.layers import Dropout


# In[57]:


model = tf.keras.models.Sequential() #i/p
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))  # 1st hidden layer
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))  # 2nd hidden layer
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))  # 3rd hidden layer

model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))  # o/p

#model.add(Dropout(0.2)) # after o/p layer--> only you looks that model is over fitting

adam=tf.keras.optimizers.Adam(learning_rate=.001) # to do not miss optiment point
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[58]:


model.fit(mnist_train_x , mnist_train_y, epochs = 50, validation_split = .2, batch_size = 100)


# In[ ]:


# After every batch we are changing weight


# In[59]:


pred=model.predict(mnist_test_x)


# In[60]:


pred


# In[61]:


pred_classes=pred.argmax(axis=1)


# In[62]:


pred_classes


# In[63]:


from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,f1_score
tab=confusion_matrix(mnist_test_y,pred_classes)


# In[64]:


tab


# In[65]:


accuracy_score(mnist_test_y,pred_classes)


# In[66]:


df1=pd.DataFrame(model.history.history)
df1


# In[67]:


plt.figure(figsize=(10,10))
plt.plot(df1.loss,color='red',marker='*',label='train')
plt.plot(df1.val_loss,color='green',marker='.',label='validation')
plt.legend()


# In[ ]:





# In[68]:


# when plot goes up then it is over fitting


# In[ ]:





# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from tensorflow.keras.utils import to_categorical
import os
#cv2 (OpenCV): reading and resizing images
#tensorflow/keras: creating and training the CNN model
#os: for navigating directories


# In[70]:


#path1: directory containing images
#cate: categories or class labels (dogs and cats)
#image_size: resize all images to 200x200 pixels
#input_image: stores image data and their labels

#Iterates over dog and cat folders
#Reads and resizes each image
#Appends the image data and its corresponding label (0 or 1) to input_image


# In[72]:


path1=r"C:\Users\user\Desktop\MLA\images\Cat&Dog"
cate=['dog','cat']
image_size=200
input_image=[]
for i in cate:
    folders=os.path.join(path1,i)
    label=cate.index(i)# tell software which image is cat and which one is dog-->  # 0 for dog, 1 for cat
    for image in os.listdir(folders):
        image_path=os.path.join(folders,image)
        image_array=cv2.imread(image_path) # using cv2 iam reading the image and storing in
        image_array=cv2.resize(image_array,(image_size,image_size))
        input_image.append([image_array,label ])



#Resizes the image to a fixed shape, making sure all images have the same dimensions, which is essential for feeding into a neural network.
# Details:
#image_array: the original image read using cv2.imread().
#image_size: a variable (e.g., 200), so the image becomes 200×200 pixels.
#The shape becomes (200, 200, 3) for RGB images (3 channels: Red, Green, Blue).



# In[73]:


len(input_image)


# In[74]:


# suffling
np.random.shuffle(input_image)


# In[75]:


x=[]
y=[]

for x_values, label in input_image:
    x.append(x_values)
    y.append(label)


# In[76]:


label
#x_values	numpy.ndarray	(200, 200, 3)	RGB image data
#label	int	()	0 (dog) or 1 (cat)


# In[77]:


x=np.array(x)
y=np.array(y)


# In[91]:


plt.imshow(x[11])


# In[79]:


#y[0] is not image data. It's a label:
#0 → dog
#1 → cat.imshow(y[0])


# In[80]:


x=x/255
#Scales pixel values from [0,255] to [0,1] for faster convergence in neural networks
#The pixel values range from 0–255 before normalization, and 0–1 after normalization (x = x / 255).


# In[81]:


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu',))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())  
model.add(Dense(128, activation='relu',input_shape=x.shape[1:]))
#model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))

#Conv2D: convolutional layer with 16 filters and 3x3 kernel
#MaxPool2D: reduces the spatial dimensions
#Flatten: flattens the output of conv layers for dense layers
#Dense(128): fully connected layer with 128 neurons and ReLU
#Dense(2): output layer with 2 neurons (dog/cat), softmax for classification


# In[ ]:





# In[82]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Adam: optimization algorithm
#sparse_categorical_crossentropy: suitable when labels are integers (0, 1)
#accuracy: performance metric


# In[83]:


model.fit(x,y,epochs=20,validation_split=0.002)

#Trains the model for 20 epochs
#validation_split=0.002 keeps 0.2% of data for validation


# In[85]:


df=pd.DataFrame(model.history.history)
df


# In[86]:


plt.figure(figsize=(12,7))
plt.plot(df.loss,color='red',label='Train',marker="*")
plt.plot(df.val_loss,color='blue',label='Validation',marker="*")
plt.legend();


# In[ ]:





# In[ ]:





# # RNN

# ## Trip_Advisor
# 

# In[ ]:


import pandas as pd


# In[ ]:


tp=pd.read_csv(r"C:\Users\sudha\OneDrive\Desktop\files\Files\Trip_advisor_review.csv", encoding='latin1')


# In[ ]:


tp.shape


# In[ ]:


tp.head()


# In[ ]:


tp.Rating.value_counts()


# In[ ]:


tp.Rating.replace({1:0,2:0,3:1,4:2,5:2},inplace=True)


# In[ ]:


tp.Rating.value_counts()

#0 → Negative (1 or 2)
#1 → Neutral (3)
#2 → Positive (4 or 5)


# In[ ]:


tp_x = tp.iloc[::,0]
tp_y = tp.iloc[::,1]


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(tp_x , tp_y ,test_size = 0.2)


# In[ ]:


from tensorflow.keras.utils import to_categorical


# In[ ]:


y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#One-Hot Encoding the Labels
#Converts the numeric labels (0, 1, 2) into one-hot vectors:



# In[ ]:


y_train.shape


# In[ ]:


import tensorflow as tf
import keras
from keras.models import Sequential


# In[ ]:


max_num_words=12000             ## from the entire corpus select 10000 words
seq_len=100                    ## how many words out of 10000 you wish to tpke from 1 doc
embedding_size=200              ## vector length of each word


# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences


# In[ ]:


tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(tp.Review)  #----># Fit on all reviews

x_train=tokenizer.texts_to_sequences(x_train)
x_test=tokenizer.texts_to_sequences(x_test)

# Tokenization and Padding
#Tokenizer: Converts words into integer sequences.


# In[ ]:


len(x_train[0])


# In[ ]:


len(x_train[11])


# In[ ]:


from tensorflow.keras.layers import Embedding,Dropout
from keras.models import Sequential


# In[ ]:


x_train=pad_sequences(x_train,maxlen=seq_len)
x_test=pad_sequences(x_test,maxlen=seq_len)

model=Sequential()
model.add(Embedding(input_dim=max_num_words,
                   input_length=seq_len,
                   output_dim=embedding_size))


##pad_sequences: Ensures all sequences are of equal length (seq_len=100).
#Makes sure every review is exactly 100 words long (pads or truncates).



# In[ ]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense


# In[ ]:


model = Sequential()
model.add(Embedding(input_dim=max_num_words, output_dim=embedding_size, input_length=seq_len))
model.add(SimpleRNN(units=64))  # Recurrent layer
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax')) #3 output classes

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


#Embedding: Turns integer sequences into dense vectors.
#SimpleRNN: Recurrent Neural Network layer to capture word sequences.
#Dense: Output layer with softmax for multi-class classification.


# In[ ]:


model.fit(x_train, y_train, validation_split=0.2, batch_size=32, epochs=5)


# In[ ]:


pred_test = model.predict(x_test)
pred_classes_test = pred_test.argmax(axis=1)
y_test_actual = y_test.argmax(axis=1)

#argmax(axis=1) returns the index of the highest value along each row.
#Converts the softmax probability vector to a predicted class label (0, 1, or 2)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[ ]:


conf_matrix_test = confusion_matrix(y_test_actual, pred_classes_test)
conf_matrix_test


# In[ ]:


accuracy_score(y_test_actual, pred_classes_test) * 100


# In[ ]:





# In[ ]:


pred_train = model.predict(x_train)
pred_classes_train = pred_train.argmax(axis=1)
y_train_actual = y_train.argmax(axis=1)

conf_matrix_train = confusion_matrix(y_train_actual, pred_classes_train)
accuracy_score(y_train_actual, pred_classes_train) * 100


# ### Data set
# ##### 1)fake_job_postings 2)Trip_advisor_review 3)spam1 4)Reviews 5)emotion_nlp

# In[ ]:





# In[ ]:





# # LSTM

# In[ ]:


import pandas as pd 


# In[ ]:


sp=pd.read_csv(r"C:\Users\sudha\OneDrive\Desktop\files\Files\spam1.csv",encoding='latin1')


# In[ ]:


sp.shape


# In[ ]:


sp.head()


# In[ ]:


sp=sp.loc[:,['v1','v2']]
sp.head()


# In[ ]:


sp=sp.rename(columns={'v1':'y','v2':'x'})
sp.head()


# In[ ]:


sp.y.replace({'spam':1,'ham':0},inplace=True)
sp.head()


# In[ ]:


sp.isnull().sum()[sp.isnull().sum()>0]


# In[ ]:


sp.x=sp.x.str.lower()
sp.head()


# In[ ]:


sp_x=sp.iloc[:,1]
sp_y=sp.iloc[:,0]


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(sp_x,sp_y,test_size=0.2)


# In[ ]:


from tensorflow.keras.utils import to_categorical


# In[ ]:


y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


# In[ ]:


# document one recrod is one document 
# corpus collection of documents 


# In[ ]:


max_num_words=10000 #from the entire corpus select a particular no of words say 10,000
seq_len=50  #how many words outta 10000 you wish to take from each document 
embedding_size=100 #vector length of each word 


# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#from keras.preprocessing.text import Tokenizer # to assign  number to words
#from keras.preprocessing.sequence import pad_sequences # like padding in cnn add zeroes to make length of
# data uniform in starting 


# In[ ]:


tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(sp.x)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[ ]:


x_train=pad_sequences(x_train,maxlen=seq_len)
x_test=pad_sequences(x_test,maxlen=seq_len)
model=Sequential()
model.add(Embedding(input_dim=max_num_words,
         input_length=seq_len,
         output_dim=embedding_size))


# In[ ]:


from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam


model.add(LSTM(32))
model.add(Dense(2,activation='softmax'))
adam=Adam(learning_rate=0.01)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


model.fit(x_train,y_train,epochs=8,batch_size=32,validation_split=0.2)


# In[ ]:


pred=model.predict(x_test)
pred


# In[ ]:


pred_classes=pred.argmax(axis=1)
pred_classes


# In[ ]:


y_test=y_test.argmax(axis=1)


# In[ ]:


pred_classes


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[ ]:


a=confusion_matrix(y_test,pred_classes)
a


# In[ ]:


print(classification_report(y_test,pred_classes))


# In[ ]:


accuracy_score(y_test,pred_classes)*100


# In[ ]:





# In[ ]:


import seaborn as sns


# In[ ]:


sns.heatmap(a, annot=True, fmt='d', square=True, cmap='coolwarm')


# In[ ]:





# # Bidirectional LSTM

# In[ ]:


from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam


# In[ ]:


model = Sequential()
model.add(Embedding(input_dim=max_num_words,
                    input_length=seq_len,
                    output_dim=embedding_size))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(2, activation='softmax'))

adam = Adam(learning_rate=0.01)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=8, batch_size=32, validation_split=0.2)


# In[ ]:


pred=model.predict(x_test)
pred


# In[ ]:


pred_classes=pred.argmax(axis=1)


# In[ ]:


y_test = to_categorical(y_test)


# In[ ]:


y_test=y_test.argmax(axis=1)


# In[ ]:


pred_classes


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[ ]:


a=confusion_matrix(y_test,pred_classes)
a


# In[ ]:


print(classification_report(y_test,pred_classes))


# In[ ]:


accuracy_score(y_test,pred_classes)*100


# In[ ]:





# # Bidirectional LSTM

# In[ ]:


import pandas as pd 


# In[ ]:


sp=pd.read_csv(r"C:\Users\sudha\OneDrive\Desktop\files\Files\spam1.csv",encoding='latin1')


# In[ ]:


sp.shape


# In[ ]:


sp.head()


# In[ ]:


sp=sp.loc[:,['v1','v2']]
sp.head()


# In[ ]:


sp=sp.rename(columns={'v1':'y','v2':'x'})
sp.head()


# In[ ]:


sp.y.replace({'spam':1,'ham':0},inplace=True)
sp.head()


# In[ ]:


sp.isnull().sum()[sp.isnull().sum()>0]


# In[ ]:


sp.x=sp.x.str.lower()
sp.head()


# In[ ]:


sp_x=sp.iloc[:,1]
sp_y=sp.iloc[:,0]


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(sp_x,sp_y,test_size=0.2)


# In[ ]:


from tensorflow.keras.utils import to_categorical


# In[ ]:


y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


# In[ ]:


# document one recrod is one document 
# corpus collection of documents 


# In[ ]:


max_num_words=10000 #from the entire corpus select a particular no of words say 10,000
seq_len=50  #how many words outta 10000 you wish to take from each document 
embedding_size=100 #vector length of each word 


# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#from keras.preprocessing.text import Tokenizer # to assign  number to words
#from keras.preprocessing.sequence import pad_sequences # like padding in cnn add zeroes to make length of
# data uniform in starting 


# In[ ]:


tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(sp.x)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[ ]:


x_train=pad_sequences(x_train,maxlen=seq_len)
x_test=pad_sequences(x_test,maxlen=seq_len)
model=Sequential()
model.add(Embedding(input_dim=max_num_words,
         input_length=seq_len,
         output_dim=embedding_size))


# In[ ]:


from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from keras.layers import Bidirectional


# In[ ]:


model.add(Bidirectional(LSTM(64)))
model.add(Dense(2,activation='softmax'))
adam=Adam(learning_rate=.001)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:





# In[ ]:


model.fit(x_train,y_train,epochs=5,batch_size=32,validation_split=0.2)


# In[ ]:


pred=model.predict(x_test)
pred


# In[ ]:


pred_classes=pred.argmax(axis=1)


# In[ ]:


y_test=y_test.argmax(axis=1)


# In[ ]:


pred_classes


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[ ]:


a=confusion_matrix(y_test,pred_classes)
a


# In[ ]:


print(classification_report(y_test,pred_classes))


# In[ ]:


accuracy_score(y_test,pred_classes)*100


# In[ ]:





# # emotion_nlp

# In[ ]:


import pandas as pd


# In[ ]:


pd.read_csv(r'C:\Users\sudha\OneDrive\Desktop\files\Files\emotion_nlp.csv')


# In[ ]:


em=pd.read_csv(r'C:\Users\sudha\OneDrive\Desktop\files\Files\emotion_nlp.csv')


# In[ ]:


em.head()


# In[ ]:


em.Emotions.value_counts()


# In[ ]:


em.isnull().sum()[em.isnull().sum()>0]


# In[ ]:


em.Emotions.unique()


# In[ ]:


em.rename(columns={'Emotions':'y','Text':'x'},inplace=True)


# In[ ]:


em.replace({'sadness':0,'anger':0,'love':1,'surprise':1,'fear':2,'joy':3},inplace=True)


# In[ ]:


em.x=em.x.str.lower() 


# In[ ]:


em


# In[ ]:


em.y.value_counts()


# In[ ]:


em_x =  em.iloc[:,0]
em_y = em.iloc[:,1]
from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(em_x, em_y, test_size=.2)


# In[ ]:


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train) # one hot endcoding


# In[ ]:


max_num_words = 10000      # from the entire corpus select 10000 words
seq_len = 50               # how many words out of 10000 you wish to take from each document
embeddings_size = 100      # vector length of each word


# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences


# In[ ]:


tokenizer = Tokenizer(num_words = max_num_words)
tokenizer.fit_on_texts(em.x)
x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, maxlen=seq_len)


x_test = pad_sequences(x_test, maxlen=seq_len)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[ ]:


model = Sequential() # initialize the network
model.add(Embedding(input_dim= max_num_words,
                   input_length= seq_len,
                   output_dim = embeddings_size))


# In[ ]:


from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from keras.layers import Bidirectional,Dense,Embedding


# In[ ]:


model = Sequential()
model.add(Embedding(input_dim=max_num_words, output_dim=embeddings_size, input_length=seq_len))
model.add(Bidirectional(LSTM(65)))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(x_train, y_train, epochs = 5, batch_size = 32, validation_split=.2)


# In[ ]:


pred=model.predict(x_test)
pred


# In[ ]:


pred_classes=pred.argmax(axis=1)


# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[ ]:


tab=confusion_matrix(y_test,pred_classes)
tab


# In[ ]:


accuracy_score(y_test,pred_classes)


# In[ ]:





# In[ ]:





# In[ ]:




