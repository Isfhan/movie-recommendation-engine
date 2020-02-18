import pandas as pd
import numpy as numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title_from_index(index):
    return df[df.index == index]['title'].values[0]

def get_index_from_title(title):
    return df[df.title == title]['index'].values[0]

# we are going to make content base movie recommender

# read csv file
df = pd.read_csv("./data/movie_dataset.csv")
# it will print first five row of data
#print(df.head())
# it will print columns of movie_dataset
#print(df.columns)

# now we need to feature selection
features = ['keywords','cast','director','genres']

# let do some data cleaning
for feature in features:
    df[feature] = df[feature].fillna('')

# lets select these columns in data frame in pandas
def combine_features(row):
    return str(row['keywords'])+ ' '+str(row['cast'])+' '+str(row['genres'])+' '+str(row['director'])

df["combine_features"] = df.apply(combine_features,axis=1)

#print(df['combine_features'].head(1))

# we are trying to find how much the value of data are similer
# here we are create an instance of CountVectorizer
count_vectorizer = CountVectorizer()

# this function retun the count of repate words
count_vectorizer_transfomed = count_vectorizer.fit_transform(df["combine_features"])
#print(count_vectorizer_transfomed.toarray())
similarity_score = cosine_similarity(count_vectorizer_transfomed)

#print(similarity_score)

movie_user_likes = "Sunshine"

movie_index = get_index_from_title(movie_user_likes)

similer_movies = list(enumerate(similarity_score[movie_index]))

sorted_similer_movies = sorted(similer_movies,key= lambda x:x[1],reverse=True)

count = 0
for movie in sorted_similer_movies:
    print(get_title_from_index(movie[0]))
    count = count+1
    if count > 20:
        break