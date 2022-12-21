#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importuri:
import os
import numpy as np


# In[2]:


#citirea datelor:
import pandas as pd
data_path = ''
train_data_df = pd.read_csv(os.path.join(data_path, 'train_data.csv'))


# In[3]:


print(train_data_df)


# In[4]:


print('Distributia etichetelor in datele de antrenare \n', train_data_df['label'].value_counts())


# In[5]:


print(train_data_df['language'].value_counts())


# In[6]:


#codificarea etichetelor din string in int:
etichete_unice = train_data_df['label'].unique()
print(etichete_unice)
label2id = {}
id2label = {}
for idx, eticheta in enumerate(etichete_unice):
    label2id[eticheta]=idx
    id2label[idx]=eticheta

print(label2id)
print(id2label)


# In[7]:


labels = []
for eticheta in train_data_df['label']:
    labels.append(label2id[eticheta])
labels=np.array(labels)

print(labels[:10])


# In[8]:


#preprocesarea datelor:
import re
import nltk
from nltk.corpus import stopwords
sw_nltk = stopwords.words('dutch')+stopwords.words('italian')+stopwords.words('danish')+stopwords.words('german')+stopwords.words('spanish')
pattern = r'[0-9]'

def proceseaza(text):
    text = re.sub("[-.,;:!?\"\'\/()_*=`]", "", text)
    words = [word for word in text.split() if word.lower() not in sw_nltk]
    new_text1 = " ".join(words)
    new_text2=new_text1.lower()
    new_text3=re.sub(pattern, '', new_text2)
    text_in_cuvinte = new_text3.strip().split(' ')
    return text_in_cuvinte

# cuvintele rezultate din functia de preprocesare:
exemple_italiano = train_data_df[train_data_df['language'] == 'italiano']
print(exemple_italiano)


# In[9]:


#test functie preprocesare
text_italiano = exemple_italiano['text'].iloc[0]
print(proceseaza(text_italiano)[:13])


# In[10]:


#aplicam functia de preprocesarea intregului set de date
processed_text = []
for text in train_data_df['text']:
    processed_text.append(proceseaza(text))
print(processed_text[0])


# In[11]:


#implementam tf-idf:
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def useless_function(doc):
    return doc
tfIdfVectorizer=TfidfVectorizer(use_idf=True, analyzer='word', tokenizer=useless_function, preprocessor=useless_function, token_pattern=None)
tfIdf = tfIdfVectorizer.fit_transform(processed_text)
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
#testam implementarea:
print (df.head(25))


# In[12]:


from sklearn.model_selection import train_test_split
print(train_data_df.shape)
print(train_data_df.head())
data = tfIdf
label = labels
data_train, data_test, label_train, label_test = train_test_split(data, label, random_state=104, train_size=0.8, shuffle=True)


# In[13]:


print(f'Nr de exemple de train {len(label_train)}')
print(f'Nr de exemple de test {len(label_test)}')


# In[14]:


from sklearn.metrics import accuracy_score
from sklearn import svm

model = svm.LinearSVC(C=1)
model.fit(data_train, label_train)

tpreds = model.predict(data_test)

print(accuracy_score(label_test, tpreds))


# In[15]:


#citirea datelor:
import pandas as pd
data_path = ''
test_data_df = pd.read_csv(os.path.join(data_path, 'test_data.csv'))


# In[16]:


processed_text_test = []
for text in test_data_df['text']:
    processed_text_test.append(proceseaza(text))
print(processed_text_test[0])


# In[17]:


#implementam tf-idf:
from sklearn.feature_extraction.text import TfidfVectorizer

def useless_function(doc):
    return doc
# tfIdfVectorizer=TfidfVectorizer(use_idf=True, analyzer='word', tokenizer=useless_function, preprocessor=useless_function, token_pattern=None)
tfIdf_test = tfIdfVectorizer.transform(processed_text_test)
df = pd.DataFrame(tfIdf_test[0].T.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
#testam implementarea:
print (df.head(25))


# In[18]:


predictions = model.predict(tfIdf_test)

print(predictions)


# In[19]:


labels_string = []
for eticheta in predictions:
    labels_string.append(id2label[eticheta])
labels_string=np.array(labels_string)

print(labels_string[:10])


# In[20]:


exported = pd.DataFrame({'id': test_data_df.index+1, 'label': labels_string})
exported.to_csv('submission_kaggle_andreea.csv', index=False)


# In[ ]:




