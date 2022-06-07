#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle


# In[2]:


data=pd.read_excel("./End-to-End-Movie-Recommendation-System-with-deployment-using-docker-and-kubernetes/movie data_new.xlsx")
data.info()
data.rename(columns={'Unnamed: 0': 'movie_id'}, inplace=True)
data.rename(columns={'Movie Name': 'Title'}, inplace=True)
columns=['Cast','Director','Genre','Title','Description']
data.info()
# data[columns].isnull().values.any()#no null values


# In[3]:


data.shape


# In[4]:


def get_important_features(data):
    important_features=[]
    for i in range (0,data.shape[0]):
        important_features.append(data['Title'][i]+' '+data['Director'][i]+' '+data['Genre'][i]+' '+data['Description'][i])
    return important_features


#creating a column to hold the combined strings
data['important_features']=get_important_features(data)
data['important_features']


# In[5]:


tfidf = TfidfVectorizer(stop_words='english')
#data['Description'] = data['Description'].fillna('')
tfidf_matrix = tfidf.fit_transform(data['important_features'])
tfidf_matrix.shape


# In[6]:


print(tfidf_matrix)


# In[7]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim.shape


# In[8]:


indices = pd.Series(data.index, index=data['Title']).drop_duplicates()
print(indices, indices.shape)


# In[9]:


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
#     print('one - ',sim_scores[:2])
    # Sort the movies based on the similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     print('two - ',sim_scores[:2])
    sim_scores = sim_scores[1:6]
#     print('three - ',sim_scores[:2])
    movie_indices = [i[0] for i in sim_scores]
#     print(movie_indices)
    # Return the top 5 most similar movies
    movies=data['Title'].iloc[movie_indices]
    id_=data['movie_id'].iloc[movie_indices]
    dict_={"Movies":movies,"id":id_}
    final_df=pd.DataFrame(dict_)
    final_df.reset_index(drop=True,inplace=True)
    return final_df


# In[10]:


print(get_recommendations('Spider-Man: Far from Home'))
#Stillwater
print(get_recommendations('Stillwater'))


# In[11]:


data.info()
new_ = data.drop(columns=['Year of Release','Watch Time','Genre','Movie Rating','Metascore of movie','Director','Cast','Votes','Description'])


# In[12]:


pickle.dump(new_,open('movie_list.pkl','wb'))
pickle.dump(cosine_sim,open('similarity.pkl','wb'))

