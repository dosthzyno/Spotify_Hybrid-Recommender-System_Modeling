#!/usr/bin/env python
# coding: utf-8

# # Modules

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from math import pi, ceil
plt.style.use('seaborn')
sns.set_style("whitegrid")

import urllib.request
from nltk.tokenize import RegexpTokenizer

import re
import random
import numpy as np
import pandas as pd
import random
import math
import itertools
import multiprocessing
from tqdm import tqdm
import logging
import pickle

from scipy.sparse.linalg import svds,eigs
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from scipy import stats

from time import time
import warnings
warnings.filterwarnings(action='ignore')

import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords


# # Data & FE

# In[2]:


track = pd.read_csv('2315_global_track_spotify.csv')
track = pd.DataFrame(track)
# track = track.rename(columns={'Unnamed: 0':'index'})
# track_t = track.drop(['artist_name', 'track_name', 'album_name', ])
print(track.shape)
track.head()


# In[3]:


def cleanText(readData):
    text = re.sub('[-=+#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', readData) # 쉼표(,) 뺌
 
    return text


# In[4]:


genre = []
for i in track['artist_genre']:
    if i == '[]':
        i = 'NA'
        genre.append(i.strip()) #"'[]'"
    else:
        i = cleanText(i)
        genre.append(i.strip())
track['genre'] = genre


# In[5]:


track = track[track['genre'] != "NA"]
track = track.reset_index()
track['track_popularity'] = track['track_popularity'] / 100 


# # Consine Similarity & Euclidean Distance Functions

# In[6]:


class ContentTFIDF:
    
    def __init__(self, data):
        self.data = data

    def calculateTFIDF(self):
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2) ,stop_words=stopwords.words('english'))
        tfidf_content = tfidf.fit_transform(self.data['genre'])
        return tfidf_content


# In[7]:


def euclidean_distance(x, y):   
    return np.sqrt(np.sum((x - y) ** 2))


# In[8]:


ct = ContentTFIDF(track)
ct_tfidf = ct.calculateTFIDF()
ct_tfidf.shape


# # Content-Based Recommender

# In[9]:


######## 최종 모델 1 ########

class ContentBasedRecommender_1:
    
    def __init__(self, data, tfidf):
        self.data = data
        self.tfidf = tfidf
        
    def user_info(self):
        print("--------------------------------------------------------------------\n노래를 추천해드리기 전에 잠시 당신에 대해서 알아보겠습니다 \n--------------------------------------------------------------------")
        songs = list(self.data['track_name'].values)
        song = random.sample(songs, 5)

        total_dictionary = {}
        qs=[]
        qs.append("무슨 곡이 가장 좋아요?   1) {}  2) {}  3) {}  4) {}  5) {}".format(song[0],song[1],song[2],song[3],song[4]))
        qs.append("어떤 분위기의 곡을 좋아하세요?   1) 밝은  2) 어두운")
        qs.append("어느정도의 속도를 가직 곡을 좋아하시나요?   1) 빠름  2) 느림")
        qs.append("고객님께서는 현재 감정이 어떤 상태이신가요?   1) 신남  2) 평범  3) 슬픔")
        qs.append("끝")

        for q in qs:
            question = q
            if question == "끝":
                break
            else:
                total_dictionary[question] = ""

        for i in total_dictionary:
            print(i)
            answer = input()
            total_dictionary[i] = answer 

        a = list(total_dictionary.items())
        self.music = a[0][1]
        self.mood = int(a[1][1])
        self.speed = int(a[2][1])
        self.emotion = int(a[3][1])
        
        return [self.music, self.mood, self.speed, self.emotion]
    
    
    def recommend_features(self, top=200):
    
        scaler = MinMaxScaler()
        index = self.data[self.data['track_name'] == self.music].index.values
        track_new = self.data[['danceability', 'energy', 'valence', 'tempo', 'acousticness']]
        track_scaled = scaler.fit_transform(track_new)
        target_index = track_scaled[index]

        euclidean = []
        for value in track_scaled:
            eu = euclidean_distance(target_index, value)
            euclidean.append(eu)

        self.data['euclidean_distance'] = euclidean
#         sim_feature_index = self.data[self.data.index != index[0]].index
#         result_features = self.data.iloc[sim_feature_index].sort_values(by='euclidean_distance', ascending=True)[:top]
        result_features = self.data.sort_values(by='euclidean_distance', ascending=True)[:top]
    #     result = track.iloc[sim_feature_index][:10]

        return result_features[['id','artist_name', 'track_name', 'euclidean_distance']]

    
    def recommend_genre(self, top=200):
        
        # TF-IDF
        tfidf = TfidfVectorizer(ngram_range=(1,2))
        tf_genre = tfidf.fit_transform(self.data.genre)

        # 코사인 유사도
        ts_genre = cosine_similarity(tf_genre, tf_genre)

        #특정 장르 정보 뽑아오기
        target_genre_index = self.data[self.data['track_name'] == self.music].index.values

        # 입력한 영화의 유사도 데이터 프레임 추가
        self.data["cos_similarity"] = ts_genre[target_genre_index, :].reshape(-1,1)
#         sim_genre_index = self.data[self.data.index != target_genre_index[0]].index
#         sim_genre = self.data.iloc[sim_genre_index].sort_values(by="cos_similarity", ascending=False)
        sim_genre = self.data.sort_values(by="cos_similarity", ascending=False)
        final_index = sim_genre.index.values[ : top]
        result_genre = self.data.iloc[final_index]

        return result_genre[['id','artist_name', 'track_name', 'cos_similarity']]

    
    def feature_genre_intersection(self, recommended_feature, recommended_genre):
        
        print("--------------------------------------------------------------\n장르 / 노래 분위기 / 노래 속도 / User 기분상태에 따라 추천을 해드리겠습니다 \n--------------------------------------------------------------")
        
        intersection = pd.merge(recommended_feature, recommended_genre, how='inner')
        similarity = intersection[['euclidean_distance', 'cos_similarity']]
        scaler = MinMaxScaler()
        scale = scaler.fit_transform(similarity)
        scale = pd.DataFrame(scale, columns=['eu_scaled', 'cos_scaled'])
        
        intersection['euclidean_scaled'] = scale['eu_scaled']
        intersection['cosine_scaled'] = scale['cos_scaled']
        intersection['ratio'] = intersection['euclidean_scaled'] + (1 - intersection['cosine_scaled'])
        result_intersection = intersection.sort_values('ratio', ascending=True)
        self.result = pd.merge(track, result_intersection, how='inner').sort_values(by='ratio')
        
        return self.result

    
    def get_genre_score(self):
#         cosine_sim_score = cosine_similarity(self.tfidf, self.tfidf)
#         target_genre_index = self.result[self.result['track_name'] == self.music].index.values
#         genre_score = cosine_sim_score[target_genre_index, :].reshape(-1, 1)
        genre_score = self.data['cos_similarity']
        return genre_score

    
    def get_mood_score(self):
        temp = pd.DataFrame(self.result['valence'])
        if self.mood == 1:
            temp['mood_score'] = temp['valence']
        else:
            temp['mood_score'] = temp['valence'].apply(lambda x: 1-x)
        return temp['mood_score']
    
    
    def get_speed_score(self):
        temp = pd.DataFrame(self.result['tempo'])
        temp['tempo_scaled'] = MinMaxScaler().fit_transform(pd.DataFrame(temp['tempo']))
        if self.speed == 1:
            temp['speed_score'] = temp['tempo_scaled']
        else:
            temp['speed_score'] = temp['tempo_scaled'].apply(lambda x: 1-x)
        return temp['speed_score']

    
    def get_emotion_score(self):
        temp = self.result[['danceability', 'energy', 'acousticness']]
        temp['danceability_scaled'] = MinMaxScaler().fit_transform((pd.DataFrame(temp['danceability'])))
        temp['acousticness_reverse'] = temp['acousticness'].apply(lambda x: 1-x)
        if self.emotion == 1:
            temp['emotion_score'] = temp.apply(lambda x: 1/3 * (x['danceability_scaled'] + x['energy'] + x['acousticness_reverse']), axis = 1)
        elif self.emotion == 2:
            temp['emotion_score'] = temp.apply(lambda x: 2/3 * (abs(x['danceability_scaled']-0.5) + abs(x['energy']-0.5) + abs(x['acousticness_reverse']-0.5)), axis = 1)
        else:
            temp['emotion_score'] = temp.apply(lambda x: 1/3 * ((1-x['danceability_scaled']) + (1-x['energy']) + (1-x['acousticness_reverse'])), axis = 1)
        return temp['emotion_score']

    def get_total_score(self, top_n = 10):
        result_df = self.result[['artist_name', 'track_name', 'album_name']]
        result_df['mood_score'] = pd.DataFrame(self.get_mood_score())
        result_df['speed_score'] = pd.DataFrame(self.get_speed_score())
        result_df['emotion_score'] = pd.DataFrame(self.get_emotion_score())
        result_df['genre_score'] = pd.DataFrame(self.get_genre_score())
        result_df['total_score'] = result_df.apply(lambda x: 1/6*(x['mood_score'] + x['speed_score'] + x['emotion_score']) + 0.5*x['genre_score'], axis = 1)
        
        return result_df.iloc[1:].sort_values(by ='total_score', ascending=False)[:top_n]
    
a = ContentBasedRecommender_1(track, ct_tfidf)
b = a.user_info()
c = a.recommend_features()
d = a.recommend_genre()
e = a.feature_genre_intersection(c,d)
f = a.get_total_score() 
f


# # K-Means Clustering Recommender (ML)

# In[80]:


df = pd.DataFrame(track)

# Consider this random sampe as the playlist of certain user
user_play = df.sample(n=10)
user_play.shape


# In[81]:


x_df = df[['danceability', 'energy', 'acousticness', 'valence', 'tempo']].values 
x_user = user_play[['danceability', 'energy','acousticness', 'valence', 'tempo']].values 
min_max_scaler = MinMaxScaler()
x_df_scaled = min_max_scaler.fit_transform(x_df)
x_user_scaled = min_max_scaler.fit_transform(x_user)

columns_scaled = ['danceability_scaled', 'energy_scaled', 'acousticness_scaled','valence_scaled', 'tempo_scaled']

df = pd.DataFrame(x_df_scaled, columns=columns_scaled)
user = pd.DataFrame(x_user_scaled, columns=columns_scaled)


# In[82]:


n_clusters = range(2,21)
ssd = []
sc = []

for n in n_clusters:
    km = KMeans(n_clusters=n, max_iter=300, n_init=10, init='k-means++', random_state=42)
    km.fit(x_df_scaled)
    preds = km.predict(x_df_scaled) 
    centers = km.cluster_centers_ 
    ssd.append(km.inertia_) 
    score = silhouette_score(x_df_scaled, preds, metric='euclidean')
    sc.append(score)
    print("Number of Clusters = {}, Silhouette Score = {}".format(n, score))


# In[83]:


plt.plot(n_clusters, sc, marker='.', markersize=12, color='red')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette score behavior over the number of clusters')
plt.show()


# ## Elbow Method

# In[84]:


for n, s in zip(n_clusters, ssd):
    print('Number of Clusters = {}, Sum of Squared Distances = {}'.format(n, s))


# In[85]:


plt.plot(n_clusters, ssd, marker='.', markersize=12)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method for optimal K')
plt.show()


# In[86]:


k=6
model = KMeans(n_clusters=k, random_state=42).fit(x_df_scaled)
pred = model.predict(x_df_scaled)
print('10 first clusters: ', model.labels_[:10])


# In[87]:


df['cluster'] = model.labels_

df['cluster'].value_counts().plot(kind='bar')
plt.xlabel('Cluster')
plt.ylabel('Amount of songs')
plt.title('Amount of songs per cluster')
plt.show()


# In[88]:


df_songs_joined = pd.concat([df,track], axis=1).set_index('cluster')

for cluster in range(k):
    display(df_songs_joined.loc[cluster, ['artist_name','track_name']].sample(frac=1).head(10))


# In[89]:


df_radar = df.groupby('cluster').mean().reset_index()
df_radar


# In[90]:


def make_radar(row, title, color, dframe, num_clusters):
    # number of variable
    categories=list(dframe)[1:]
    N = len(categories)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Initialise the radar plot
    #ax = plt.subplot(4,ceil(num_clusters/4),row+1, polar=True, )
    ax = plt.subplot(2,ceil(num_clusters/2),row+1, polar=True, )
    
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=14)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2,0.4,0.6,0.8], ["0.2","0.4","0.6","0.8"], color="grey", size=8)
    plt.ylim(0,1)

    # Ind1
    values=dframe.loc[row].drop('cluster').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=16, color=color, y=1.06)


# In[91]:


# initialize the figure
plt.figure(figsize=(24,15))
 
# Create a color palette:
my_palette = plt.cm.get_cmap("brg", len(df_radar.index))

# Create cluster name
title_list = ['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 
              'Cluster5', 'Cluster6']

# Loop to plot
for row in range(0, len(df_radar.index)):
    make_radar(row=row, title=str(df_radar['cluster'][row]) + ' : ' + title_list[row], 
               color=my_palette(row), dframe=df_radar, num_clusters=len(df_radar.index))


# In[92]:


pca = PCA(n_components=3, random_state=42)
songs_pca = pca.fit_transform(x_df_scaled)
print(pca.explained_variance_ratio_.sum())
plt.plot(pca.explained_variance_ratio_)


# In[93]:


df_pca = pd.DataFrame(songs_pca, columns=['C1', 'C2', 'C3'])
df_pca['cluster'] = model.labels_
minor_cluster = df['cluster'].value_counts().tail(1)
sampled_clusters_pca = pd.DataFrame()

for c in df_pca.cluster.unique():
    df_cluster_sampled_pca = df_pca[df_pca.cluster == c].sample(n=int(minor_cluster), random_state=42)
    sampled_clusters_pca = pd.concat([sampled_clusters_pca,df_cluster_sampled_pca], axis=0)
sampled_clusters_pca.cluster.value_counts()


# In[94]:


sns.scatterplot(x='C1', y='C2', hue='cluster', data=sampled_clusters_pca, legend="full", palette='Paired')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Clusters view using PCA')
plt.show()


# In[95]:


fig = plt.figure()
fig.suptitle('Clusters view with 3 dimensions using PCA')
ax = Axes3D(fig)

ax.scatter(df_pca['C1'], df_pca['C2'], df_pca['C3'],
           c=df_pca['cluster'], cmap='Paired')

ax.set_xlabel('C1')
ax.set_ylabel('C2')
ax.set_zlabel('C3')
plt.show()


# ## Predict which cluster the user's playlist belongs to

# In[96]:


user_pred = model.predict(x_user_scaled)
print('User Playlist clusters: ', user_pred)


# In[97]:


user_cluster = pd.DataFrame(x_user_scaled, columns=columns_scaled)
user_cluster['cluster'] = user_pred

user_cluster['cluster'].value_counts().plot(kind='bar', color='green')
plt.xlabel('Cluster')
plt.ylabel('Amount of songs')
plt.title('Amount of songs in the users clusters')
plt.show()


# In[98]:


user_play_r = user_play.reset_index(drop=True)
df_user_songs_joined = pd.concat([user_cluster,user_play_r], axis=1).set_index('cluster')
for cluster in user_cluster['cluster'].unique():
    display(df_user_songs_joined.loc[cluster, ['artist_name','track_name']].sample(frac=1))


# In[99]:


df_user_songs_joined.reset_index(inplace=True)
cluster_pct = df_user_songs_joined.cluster.value_counts(normalize=True)*20

if int(cluster_pct.round(0).sum()) < 20:
    cluster_pct[cluster_pct < 0.5] = cluster_pct[cluster_pct < 0.5] + 1.0
    
display(cluster_pct)


# In[100]:


df_user_songs_joined['cluster_pct'] = df_user_songs_joined['cluster'].apply(lambda c: cluster_pct[c])
df_user_songs_joined.drop(columns=columns_scaled, inplace=True)
df_user_songs_joined.head(3)


# In[101]:


df.isnull().sum()


# In[102]:


df_songs_joined = df_songs_joined.reset_index(drop=False)
playlist = pd.DataFrame()

for ncluster, pct in cluster_pct.items():
    songs = df_songs_joined[df_songs_joined['cluster'] == ncluster].sample(n=int(round(pct, 0)))
    playlist = pd.concat([playlist,songs], ignore_index=True)
    if len(playlist) > 20 :
        flag = 20 - len(playlist)
        playlist = playlist[:flag]
playlist[['artist_name', 'track_name', 'cluster']]


# In[103]:


user_play[['artist_name', 'track_name']]

