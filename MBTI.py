
# coding: utf-8

# In[162]:


import os
os.chdir('D:/CSUEB/BAN 675 Text mining/Group Project/mbti_1.csv')
os.getcwd()


# In[163]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ###  E or I --> Extraversion or Introversion
# ###  S or N --> Sensing or Intuition
# ###  T or F --> Thinking or Feeling
# ###   J or P --> Judging or Perceiving

# In[164]:


df=pd.read_csv('mbti_1.csv') #C:/Users/Dell/Documents/Text Mining/
df.head()


# In[165]:


df.groupby('type').agg({'type':'count'})


# In[166]:


cnt_types = df['type'].value_counts()

plt.figure(figsize=(12,4))
sns.barplot(cnt_types.index, cnt_types.values) #palette="Reds_d"
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Types', fontsize=12)
plt.show()


# ### Add columns for type indicators

# In[167]:


def get_types(row):
    t=row['type']

    I = 0; N = 0
    T = 0; J = 0
    
    if t[0] == 'I': I = 1
    elif t[0] == 'E': I = 0
    else: print('I-E incorrect')
        
    if t[1] == 'N': N = 1
    elif t[1] == 'S': N = 0
    else: print('N-S incorrect')
        
    if t[2] == 'T': T = 1
    elif t[2] == 'F': T = 0
    else: print('T-F incorrect')
        
    if t[3] == 'J': J = 1
    elif t[3] == 'P': J = 0
    else: print('J-P incorrect')
    return pd.Series( {'IE':I, 'NS':N , 'TF': T, 'JP': J }) 

data = df.join(df.apply (lambda row: get_types (row),axis=1))
data.head(5)


# In[168]:


print ("Introversion (I) /  Extroversion (E):\t", data['IE'].value_counts()[0], " / ", data['IE'].value_counts()[1])
print ("Intuition (N) – Sensing (S):\t\t", data['NS'].value_counts()[0], " / ", data['NS'].value_counts()[1])
print ("Thinking (T) – Feeling (F):\t\t", data['TF'].value_counts()[0], " / ", data['TF'].value_counts()[1])
print ("Judging (J) – Perceiving (P):\t\t", data['JP'].value_counts()[0], " / ", data['JP'].value_counts()[1])


# In[169]:


N = 4
but = (data['IE'].value_counts()[0], data['NS'].value_counts()[0], data['TF'].value_counts()[0], data['JP'].value_counts()[0])
top = (data['IE'].value_counts()[1], data['NS'].value_counts()[1], data['TF'].value_counts()[1], data['JP'].value_counts()[1])

ind = np.arange(N)    # the x locations for the groups
width = 0.7      # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, but, width)
p2 = plt.bar(ind, top, width, bottom=but)

plt.ylabel('Count')
plt.title('Distribution accoss types indicators')
plt.xticks(ind, ('I/E',  'N/S', 'T/F', 'J/P',))

plt.show()


# ### Almost no correlation

# In[170]:


data[['IE','NS','TF','JP']].corr()


# In[171]:


cmap = plt.cm.RdBu
corr = data[['IE','NS','TF','JP']].corr()
plt.figure(figsize=(12,10))
plt.title('Pearson Features Correlation', size=15)
sns.heatmap(corr, cmap=cmap,  annot=True, linewidths=1)
plt.show()


# In[172]:


df["LenPre"] = df["posts"].apply(len)
sns.distplot(df["LenPre"]).set_title("Distribution of Lengths of all 50 Posts")
plt.show()


# In[173]:


def var_row(row):
    l=[]
    for i in row.split('|||'):
        l.append(len(i.split()))
    return np.var(l)

df['words_per_comment'] = df['posts'].apply(lambda x: len(x.split())/50)
df['variance_of_word_counts'] = df['posts'].apply(lambda x: var_row(x))
df.head()
        


# In[174]:


plt.figure(figsize=(10,5))
sns.swarmplot("type", "words_per_comment", data=df)
plt.show()


# In[175]:


df_2 = df[~df['type'].isin(['ESFJ','ESFP','ESTJ','ESTP'])]
df_2['http_per_comment'] = df_2['posts'].apply(lambda x: x.count('http')/50)
df_2['qmark_per_comment'] = df_2['posts'].apply(lambda x: x.count('?')/50)
df_2.head()


# In[176]:


print(df_2.groupby('type').agg({'http_per_comment': 'mean'}))
print(df_2.groupby('type').agg({'qmark_per_comment': 'mean'}))


# In[177]:


plt.figure(figsize=(15,10))
sns.jointplot("variance_of_word_counts", "words_per_comment", data=df_2, kind="hex")
plt.show()


# In[204]:


from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS

fig, ax = plt.subplots(len(df['type'].unique()), sharex=True, figsize=(15,10*len(df['type'].unique())))

k = 0
for i in df['type'].unique():
    df_4 = df[df['type'] == i]
    wordcloud = WordCloud().generate(df_4['posts'].to_string())
    ax[k].imshow(wordcloud)
    ax[k].set_title(i)
    ax[k].axis("off")
    k+=1
    
plt.show()


# ### PreProcessing

# In[205]:


map1 = {"I": 0, "E": 1}
map2 = {"N": 0, "S": 1}
map3 = {"T": 0, "F": 1}
map4 = {"J": 0, "P": 1}
df['I-E'] = df['type'].astype(str).str[0]
df['I-E'] = df['I-E'].map(map1)
df['N-S'] = df['type'].astype(str).str[1]
df['N-S'] = df['N-S'].map(map2)
df['T-F'] = df['type'].astype(str).str[2]
df['T-F'] = df['T-F'].map(map3)
df['J-P'] = df['type'].astype(str).str[3]
df['J-P'] = df['J-P'].map(map4)
print(df.head(10))


# ### Test-Train split

# In[197]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np

XX = df.drop(['type','posts','I-E'], axis=1).values  
le = LabelEncoder()

yy = le.fit_transform(df['I-E'].values)

print(yy.shape)
print(XX.shape)

XX_train,XX_test,yy_train,yy_test=train_test_split(XX,yy,test_size = 0.1, random_state=5)


# ### Applying Logistic Regression

# In[211]:


from sklearn.linear_model import LogisticRegression
# Logistic Regression
logregg = LogisticRegression()
logregg.fit(XX_train, yy_train)

Y_predd = logregg.predict(XX_test)

acc_logg = round(logregg.score(XX_train, yy_train) * 100, 2)
print(round(acc_logg,2,), "%")


# ### Applying Naive Bayes

# In[209]:


yp_train = clf.predict(XX_train)
yp_test = clf.predict(XX_test)


# In[213]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(XX_train, yy_train)

acc_train=np.mean(yp_train == yy_train)*100
print("Train Accuracy:", round(acc_train,2),"%")

acc_test=np.mean(yp_test == yy_test)*100
print("Test Accuracy:", round(acc_test,2),"%")
print("******")
#print("Categorical Train Accuracy:", cat_accuracy(yp_train, yy_train, le))
#print("Categorical Test Accuracy:", cat_accuracy(yp_test, yy_test, le))

