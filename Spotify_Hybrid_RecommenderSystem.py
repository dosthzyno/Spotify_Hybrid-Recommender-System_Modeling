#!/usr/bin/env python
# coding: utf-8

# # Basic Setting

# In[1]:


import matplotlib.pyplot as plt

import random
import numpy as np
import pandas as pd

import sympy
from sympy import Matrix, init_printing

from scipy.sparse.linalg import svds,eigs
from sklearn.preprocessing import MinMaxScaler

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import pairwise_distances

from time import time
import warnings
warnings.filterwarnings(action='ignore')

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[2]:


track = pd.read_csv('1160_global_track_spotify.csv')
track = pd.DataFrame(track)
# track = track.rename(columns={'Unnamed: 0':'index'})
# track_t = track.drop(['artist_name', 'track_name', 'album_name', ])
print(track.shape)
track.head()


# # Feature Engineering 

# In[3]:


import re
 
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


# In[6]:


track.head()


# ## Collecting User info

# In[8]:


# import random
# songs = list(track['track_name'].values)
# song = random.sample(songs, 5)

# total_dictionary = {}
# qs=[]
# qs.append("무슨 곡이 가장 좋아요?   1) {}  2) {}  3) {}  4) {}  5) {}".format(song[0],song[1],song[2],song[3],song[4]))
# qs.append("어떤 분위기의 곡을 좋아하세요?   1) 밝은  2) 어두운")
# qs.append("어느정도의 속도를 가직 곡을 좋아하시나요?   1) 빠름  2) 느림")
# qs.append("고객님께서는 현재 감정이 어떤 상태이신가요?   1) 신남  2) 평범  3) 슬픔")
# qs.append("끝")


# for q in qs:
#     question = q
#     if question == "끝":
#         break
#     else:
#         total_dictionary[question] = ""

# for i in total_dictionary:
#     print(i)
#     answer = input()
#     total_dictionary[i] = answer 

# a = list(total_dictionary.items())
# answer_1 = a[0][1]
# answer_2 = a[1][1]
# answer_3 = a[2][1]
# answer_4 = a[3][1]


# ## Genre Recommeder

# In[7]:


def recommend_genre(data, music_name, top=200):

    # TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1,2))
    tf_genre = tfidf.fit_transform(data.genre)

    # 코사인 유사도
    ts_genre = cosine_similarity(tf_genre, tf_genre)

    #특정 장르 정보 뽑아오기
    target_genre_index = data[data['track_name'] == music_name].index.values

    # 입력한 영화의 유사도 데이터 프레임 추가
    data["cos_similarity"] = ts_genre[target_genre_index, :].reshape(-1,1)
    sim_genre_index = data[data.index != target_genre_index[0]].index
    sim_genre = data.iloc[sim_genre_index].sort_values(by="cos_similarity", ascending=False)
    final_index = sim_genre.index.values[ : top]
    result_genre = data.iloc[final_index]
    
    return result_genre[['artist_name', 'track_name', 'cos_similarity']]


# ## Genre + Artist/Track/Album name Recommeder

# In[8]:


# def recommend_genre(data, music_name, top=200):

#     # TF-IDF
#     tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2) ,stop_words=stopwords.words('english'))
#     tf_genre = tfidf.fit_transform(data['artist_name'] + ' ' +
#                                    data['track_name'] + ' ' +
#                                    data['album_name'] + ' ' +
#                                    data['genre'])

#     # 코사인 유사도
#     ts_genre = cosine_similarity(tf_genre, tf_genre)

#     #특정 장르 정보 뽑아오기
#     target_genre_index = data[data['track_name'] == music_name].index.values

#     # 입력한 영화의 유사도 데이터 프레임 추가
#     data["cos_similarity"] = ts_genre[target_genre_index, :].reshape(-1,1)
#     sim_genre_index = track[track.index != target_genre_index[0]].index
#     sim_genre = data.iloc[sim_genre_index].sort_values(by="cos_similarity", ascending=False)
#     final_index = sim_genre.index.values[ : top]
#     result_genre = data.iloc[final_index]
    
#     return result_genre[['artist_name', 'track_name', 'cos_similarity']]


# ## Features Recommender

# In[9]:


def recommend_features(track, x, top=200):
    
    scaler = MinMaxScaler()
    index = track[track['track_name'] == x].index.values
    track1 = track[['danceability', 'energy', 'valence', 'tempo', 'acousticness']]
    track_scaled = scaler.fit_transform(track1)
    target_index = track_scaled[index]
    
    euclidean = []
    for value in track_scaled:
        eu = euclidean_distance(target_index, value)
        euclidean.append(eu)
    
    track['euclidean_distance'] = euclidean
    sim_feature_index = track[track.index != index[0]].index

    result_feature = track.iloc[sim_feature_index].sort_values(by='euclidean_distance', ascending=True)[:200]
#     result = track.iloc[sim_feature_index][:10]


    return result_feature[['artist_name', 'track_name', 'euclidean_distance']]


# ## Intersection of Genre & Feature

# In[10]:


def feature_genre_intersection(a, b):
    intersected = pd.merge(a, b, how='inner')
    similarity = intersected[['euclidean_distance', 'cos_similarity']]
    temp =scaler.fit_transform(similarity)
    temp = pd.DataFrame(temp, columns = ['euclidean_scaled', 'cosine_scaled'])

    intersected['euclidean_scaled'] = temp['euclidean_scaled']
    intersected['cosine_scaled'] = temp['cosine_scaled']

    intersected['ratio'] = intersected['euclidean_scaled'] + (1 - intersected['cosine_scaled'])
    result = intersected.sort_values('ratio', ascending=True)[:10]
    return result


# ## Cosine Similarity and Euclidean Distance

# In[11]:


class ContentTFIDF:
    
    def __init__(self, data):
        self.data = data

    def calculateTFIDF(self):
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2) ,stop_words=stopwords.words('english'))
        tfidf_content = tfidf.fit_transform(self.data['genre'])
        return tfidf_content


# In[12]:


def euclidean_distance(x, y):   
    return np.sqrt(np.sum((x - y) ** 2))


# In[13]:


ct = ContentTFIDF(track)
ct_tfidf = ct.calculateTFIDF()
ct_tfidf.shape


# # Content-Based Recommeder
# - When a User first signs up on website

# In[80]:


# 진호
# class ContentBasedRecommender_1:
    
#     def __init__(self, data):
#         self.data = data
#         self.music = ''
#         self.mood = ''
#         self.speed = ''
#         self.emotion = ''

        
#     def user_info(self):
#         print("--------------------------------------------------------------------\n\
# 노래를 추천해드리기 전에 잠시 당신에 대해서 알아보겠습니다 \n\
# --------------------------------------------------------------------")
#         songs = list(self.data['track_name'].values)
#         song = random.sample(songs, 5)

#         total_dictionary = {}
#         qs=[]
#         qs.append("무슨 곡이 가장 좋아요?   1) {}  2) {}  3) {}  4) {}  5) {}".format(song[0],song[1],song[2],song[3],song[4]))
#         qs.append("어떤 분위기의 곡을 좋아하세요?   1) 밝은  2) 어두운")
#         qs.append("어느정도의 속도를 가직 곡을 좋아하시나요?   1) 빠름  2) 느림")
#         qs.append("고객님께서는 현재 감정이 어떤 상태이신가요?   1) 신남  2) 평범  3) 슬픔")
#         qs.append("끝")

#         for q in qs:
#             question = q
#             if question == "끝":
#                 break
#             else:
#                 total_dictionary[question] = ""

#         for i in total_dictionary:
#             print(i)
#             answer = input()
#             total_dictionary[i] = answer 

#         a = list(total_dictionary.items())
#         self.music = a[0][1]
#         self.mood = a[1][1]
#         self.speed = a[2][1]
#         self.emotion = a[3][1]
        
#         return [self.music, self.mood, self.speed, self.emotion]
    
    
#     def recommend_features(self, music_name, top=200):
    
#         scaler = MinMaxScaler()
#         index = self.data[self.data['track_name'] == music_name].index.values
#         track_new = self.data[['danceability', 'energy', 'valence', 'tempo', 'acousticness']]
#         track_scaled = scaler.fit_transform(track_new)
#         target_index = track_scaled[index]

#         euclidean = []
#         for value in track_scaled:
#             eu = euclidean_distance(target_index, value)
#             euclidean.append(eu)

#         self.data['euclidean_distance'] = euclidean
#         sim_feature_index = self.data[self.data.index != index[0]].index

#         result_features = self.data.iloc[sim_feature_index].sort_values(by='euclidean_distance', ascending=True)[:top]
#     #     result = track.iloc[sim_feature_index][:10]

#         return result_features[['artist_name', 'track_name', 'euclidean_distance']]

    
#     def recommend_genre(self, music_name, top=200):
        
#         # TF-IDF
#         tfidf = TfidfVectorizer(ngram_range=(1,2))
#         tf_genre = tfidf.fit_transform(self.data.genre)

#         # 코사인 유사도
#         ts_genre = cosine_similarity(tf_genre, tf_genre)

#         #특정 장르 정보 뽑아오기
#         target_genre_index = self.data[self.data['track_name'] == music_name].index.values

#         # 입력한 영화의 유사도 데이터 프레임 추가
#         self.data["cos_similarity"] = ts_genre[target_genre_index, :].reshape(-1,1)
#         sim_genre_index = self.data[self.data.index != target_genre_index[0]].index
#         sim_genre = self.data.iloc[sim_genre_index].sort_values(by="cos_similarity", ascending=False)
#         final_index = sim_genre.index.values[ : top]
#         result_genre = self.data.iloc[final_index]

#         return result_genre[['artist_name', 'track_name', 'cos_similarity']]

    
#     def feature_genre_intersection(self, recommended_feature, recommended_genre, top=10):
        
#         print("--------------------------------------------------------------\n\
# 장르 / 노래 분위기 / 노래 속도 / User 기분상태에 따라 추천을 해드리겠습니다 \n\
# --------------------------------------------------------------")
        
#         intersection = pd.merge(recommended_feature, recommended_genre, how='inner')
#         similarity = intersection[['euclidean_distance', 'cos_similarity']]
#         scaler = MinMaxScaler()
#         scale = scaler.fit_transform(similarity)
#         scale = pd.DataFrame(scale, columns=['eu_scaled', 'cos_scaled'])
        
#         intersection['euclidean_scaled'] = scale['eu_scaled']
#         intersection['cosine_scaled'] = scale['cos_scaled']
#         intersection['ratio'] = intersection['euclidean_scaled'] + (1 - intersection['cosine_scaled'])
#         result_intersection = intersection.sort_values('ratio', ascending=True)
        
#         return result_intersection[:top]

    
# a = ContentBasedRecommender_1(track)
# b = a.user_info()
# c = a.recommend_features(b[0])
# d = a.recommend_genre(b[0])
# e = a.feature_genre_intersection(c,d)
# e


# In[26]:


# 연재
# class ContentBasedRecommender_1:
    
#     def __init__(self, data, tfidf_matrix):
#         self.data = data
#         self.tfidf_matrix = tfidf_matrix
#         self.music = ''
#         self.mood = ''
#         self.speed = ''
#         self.emotion = ''

        
#     def user_info(self):
#         print("--------------------------------------------------------------------\n\
# 노래를 추천해드리기 전에 잠시 당신에 대해서 알아보겠습니다 \n\
# --------------------------------------------------------------------")
#         songs = list(self.data['track_name'].values)
#         song = random.sample(songs, 5)

#         total_dictionary = {}
#         qs=[]
#         qs.append("무슨 곡이 가장 좋아요?   1) {}  2) {}  3) {}  4) {}  5) {}".format(song[0],song[1],song[2],song[3],song[4]))
#         qs.append("어떤 분위기의 곡을 좋아하세요?   1) 밝은  2) 어두운")
#         qs.append("어느정도의 속도를 가직 곡을 좋아하시나요?   1) 빠름  2) 느림")
#         qs.append("고객님께서는 현재 감정이 어떤 상태이신가요?   1) 신남  2) 평범  3) 슬픔")
#         qs.append("끝")

#         for q in qs:
#             question = q
#             if question == "끝":
#                 break
#             else:
#                 total_dictionary[question] = ""

#         for i in total_dictionary:
#             print(i)
#             answer = input()
#             total_dictionary[i] = answer 

#         a = list(total_dictionary.items())
#         self.music = a[0][1]
#         self.mood = int(a[1][1])
#         self.speed = int(a[2][1])
#         self.emotion = int(a[3][1])
        
#         return [self.music, self.mood, self.speed, self.emotion]

#     def get_genre_score(self):
#         cosine_sim_score = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
#         target_genre_index = self.data[self.data['track_name'] == self.music].index.values
#         genre_score = cosine_sim_score[target_genre_index, :].reshape(-1, 1)
#         return genre_score

#     def get_mood_score(self):
#         temp = pd.DataFrame(self.data['valence'])
#         if self.mood == 1:
#             temp['mood_score'] = temp['valence']
#         else:
#             temp['mood_score'] = temp['valence'].apply(lambda x: 1-x)
#         return temp['mood_score']
    
#     def get_speed_score(self):
#         temp = pd.DataFrame(self.data['tempo'])
#         temp['tempo_scaled'] = MinMaxScaler().fit_transform(pd.DataFrame(temp['tempo']))
#         if self.speed == 1:
#             temp['speed_score'] = temp['tempo_scaled']
#         else:
#             temp['speed_score'] = temp['tempo_scaled'].apply(lambda x: 1-x)
#         return temp['speed_score']

#     def get_emotion_score(self):
#         temp = self.data[['danceability', 'energy', 'acousticness']]
#         temp['danceability_scaled'] = MinMaxScaler().fit_transform((pd.DataFrame(temp['danceability'])))
#         temp['acousticness_reverse'] = temp['acousticness'].apply(lambda x: 1-x)
#         if self.emotion == 1:
#             temp['emotion_score'] = temp.apply(lambda x: 1/3 * (x['danceability_scaled'] + x['energy'] + x['acousticness_reverse']), axis = 1)
#         elif self.emotion == 2:
#             temp['emotion_score'] = temp.apply(lambda x: 2/3 * (abs(x['danceability_scaled']-0.5) + abs(x['energy']-0.5) + abs(x['acousticness_reverse']-0.5)), axis = 1)
#         else:
#             temp['emotion_score'] = temp.apply(lambda x: 1/3 * ((1-x['danceability_scaled']) + (1-x['energy']) + (1-x['acousticness_reverse'])), axis = 1)
#         return temp['emotion_score']

#     def get_total_score(self, top_n = 20):
#         result_df = self.data[['artist_name', 'track_name', 'album_name']]
#         result_df['mood_score'] = pd.DataFrame(self.get_mood_score())
#         result_df['speed_score'] = pd.DataFrame(self.get_speed_score())
#         result_df['emotion_score'] = pd.DataFrame(self.get_emotion_score())
#         result_df['genre_score'] = pd.DataFrame(self.get_genre_score())
#         result_df['total_score'] = result_df.apply(lambda x: 1/6*(x['mood_score'] + x['speed_score'] + x['emotion_score']) + 0.5*x['genre_score'], axis = 1)
        
#         target_genre_index = self.data[self.data['track_name'] == self.music].index.values
        
#         return result_df.iloc[result_df.index != target_genre_index[0]].sort_values(by = 'total_score', ascending=False)[:top_n]

# cbr = ContentBasedRecommender_1(track, ct_tfidf)
# ui1 = cbr.user_info()
# ex1 = cbr.get_total_score()
# ex1


# In[79]:


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
        self.mood = a[1][1]
        self.speed = a[2][1]
        self.emotion = a[3][1]
        
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
        cosine_sim_score = cosine_similarity(self.tfidf, self.tfidf)
        target_genre_index = self.result[self.result['track_name'] == self.music].index.values
        genre_score = cosine_sim_score[target_genre_index, :].reshape(-1, 1)
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

    def get_total_score(self, top_n = 20):
        result_df = self.result[['artist_name', 'track_name', 'album_name']]
        result_df['mood_score'] = pd.DataFrame(self.get_mood_score())
        result_df['speed_score'] = pd.DataFrame(self.get_speed_score())
        result_df['emotion_score'] = pd.DataFrame(self.get_emotion_score())
        result_df['genre_score'] = pd.DataFrame(self.get_genre_score())
        result_df['total_score'] = result_df.apply(lambda x: 1/6*(x['mood_score'] + x['speed_score'] + x['emotion_score']) + 0.5*x['genre_score'], axis = 1)
        
        return result_df.iloc[1:].sort_values(by = 'total_score', ascending=False)[:top_n]
    
a = ContentBasedRecommender_1(track, ct_tfidf)
b = a.user_info()
c = a.recommend_features()
d = a.recommend_genre()
e = a.feature_genre_intersection(c,d)
f = a.get_total_score()
f                                                                               


# In[36]:


track.iloc[e.index.values]


# # Content-Based Recommender 
# - When website does not have enough users
# - 유저의 플레이 리스트에서 곡을 전부 따옴 
# - 유저의 플레이 리스트 기반으로 장르(코사인 유사도) / 피쳐(유클리디안 거리)를 구함
# - 코사인 유사도가 높으면서 유클리디안 거리가 가까운것을 기준으로 SCORE를 만들어서 추천

# In[156]:


# 일단 유저의 playlist가 없으니 랜덤으로 우리 track에서 20개정도 뽑은걸 user의 playlist라 가정하겠음
user_play = track.sample(n=20)
user_play.shape


# In[ ]:


class ContentBasedRecommender_2:
    
    def __init__(self, user_playlist):
        self.data = user_playlist
        self.music = ''
        self.mood = ''
        self.speed = ''
        self.emotion = ''

        
    def euclidean_distance(self, x, y):   
        return np.sqrt(np.sum((x - y) ** 2))
    
    
    def recommend_features(self, music_name, top=200):
    
        scaler = MinMaxScaler()
        index = self.data[self.data['track_name'] == music_name].index.values
        track_new = self.data[['danceability', 'energy', 'valence', 'tempo', 'acousticness']]
        track_scaled = scaler.fit_transform(track_new)
        target_index = track_scaled[index]

        euclidean = []
        for value in track_scaled:
            eu = euclidean_distance(target_index, value)
            euclidean.append(eu)

        self.data['euclidean_distance'] = euclidean
        sim_feature_index = self.data[self.data.index != index[0]].index

        result_features = self.data.iloc[sim_feature_index].sort_values(by='euclidean_distance', ascending=True)[:top]
    #     result = track.iloc[sim_feature_index][:10]

        return result_features[['artist_name', 'track_name', 'euclidean_distance']]

    
    def recommend_genre(self, music_name, top=200):
        
        # TF-IDF
        tfidf = TfidfVectorizer(ngram_range=(1,2))
        tf_genre = tfidf.fit_transform(self.data.genre)

        # 코사인 유사도
        ts_genre = cosine_similarity(tf_genre, tf_genre)

        #특정 장르 정보 뽑아오기
        target_genre_index = self.data[self.data['track_name'] == music_name].index.values

        # 입력한 영화의 유사도 데이터 프레임 추가
        self.data["cos_similarity"] = ts_genre[target_genre_index, :].reshape(-1,1)
        sim_genre_index = self.data[self.data.index != target_genre_index[0]].index
        sim_genre = self.data.iloc[sim_genre_index].sort_values(by="cos_similarity", ascending=False)
        final_index = sim_genre.index.values[ : top]
        result_genre = self.data.iloc[final_index]

        return result_genre[['artist_name', 'track_name', 'cos_similarity']]

    
    def feature_genre_intersection(self, recommended_feature, recommended_genre, top=10):
        
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
        
        return result_intersection[:top]

    
a = ContentBasedRecommender_1(track)
b = a.user_info()
c = a.recommend_features(b[0])
d = a.recommend_genre(b[0])
e = a.feature_genre_intersection(c,d)
e


# In[215]:


def recommend_features(data, user_playlist, top=200):

    scaler = MinMaxScaler()
    user_in_track = []
    for music in user_playlist['id']:
        music_index = track[track['id'] == music].index
        user_in_track.append(music[0])
    
    track_new = data[['danceability', 'energy', 'valence', 'tempo', 'acousticness']]
    track_scaled = scaler.fit_transform(track_new)
    track_scaled = pd.DataFrame(track_scaled)
    target_index = track_scaled.iloc[user_in_track]
    target_index_distance = target_index.describe().iloc[1].values

    euclidean = []
    for value in range(len(track_scaled)):
        eu = euclidean_distance(target_index_distance, track_scaled.iloc[value])
        euclidean.append(eu)

    data['euclidean_distance'] = euclidean
#     sim_feature_index = data[data.index != target_index.values].index

    result_features = data.sort_values(by='euclidean_distance', ascending=True)
#     result = track.iloc[sim_feature_index][:10]

    return result_features[['artist_name', 'track_name', 'euclidean_distance']][:top]


# In[216]:


recommend_features(track,user_play)


# In[218]:


user_play[['artist_name', 'track_name']]


# In[204]:


scaler = MinMaxScaler()
user_in_track = []
for i in user_play['id']:
    b = track[track['id'] == i].index.values
    user_in_track.append(b[0])
user_in_track
track.iloc[user_in_track]


track_new = track[['danceability', 'energy', 'valence', 'tempo', 'acousticness']]
track_scaled = scaler.fit_transform(track_new)
track_scaled = pd.DataFrame(track_scaled)
target_index = track_scaled.iloc[user_in_track]
target_index_distance = target_index.describe().iloc[1].values
euclidean = []
for value in range(len(track_scaled)):
    eu = euclidean_distance(target_index_distance, track_scaled.iloc[value])
    euclidean.append(eu)
euclidean
# target_index.describe().iloc[1].values
# track_scaled.iloc[1]
target_index[0]


# In[ ]:





# # Collaborative Filtering Recommender
# - When there are enough users

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




