from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# we are trying to find how much the value of data are similer
data  = ['Karachi Pakistan Pakistan','Karachi Pakistan Karachi']

# here we are create an instance of CountVectorizer
count_vectorizer = CountVectorizer()

# this function retun the count of repate words
count_vectorizer_transfomed = count_vectorizer.fit_transform(data)
#print(count_vectorizer_transfomed.toarray())

similarity_score = cosine_similarity(count_vectorizer_transfomed)

print(similarity_score)