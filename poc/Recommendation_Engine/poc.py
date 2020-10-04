import pandas as pd
from sklearn.model_selection import train_test_split
from math import pow, sqrt

### Data related
# Reading users
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('data/users.dat', sep='::', names=u_cols, encoding='latin-1', engine='python')

# Reading ratings
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('data/ratings.dat', sep='::', names=r_cols, encoding='latin-1', engine='python')

# Reading movies dataset into a pandas dataframe object.
m_cols = ['movie_id', 'movie_title', 'genre']
movies = pd.read_csv('data/movies.dat', sep='::', names=m_cols, encoding='latin-1', engine='python')

# Getting series of lists by applying split operation.
movies.genre = movies.genre.str.split('|')
genre_columns = list(set([j for i in movies['genre'].tolist() for j in i]))

# Iterating over every list to create and fill values into columns.
for j in genre_columns:
    movies[j] = 0
for i in range(movies.shape[0]):
    for j in genre_columns:
        if(j in movies['genre'].iloc[i]):
            movies.loc[i,j] = 1

split_values = movies['movie_title'].str.split("(", n = 1, expand = True)
# Setting 'movie_title' values to title part and creating 'release_year' column.
movies.movie_title = split_values[0]
movies['release_year'] = split_values[1]
# Cleaning the release_year series and dropping 'genre' columns as it has already been one hot encoded.
movies['release_year'] = movies.release_year.str.replace(')','')
movies.drop('genre',axis=1,inplace=True)

### Implementation functions related
#Function to get the rating given by a user to a movie.
def get_rating_(userid,movieid):
    return (ratings.loc[(ratings.user_id==userid) & (ratings.movie_id == movieid),'rating'].iloc[0])

# Function to get the list of all movie ids the specified user has rated.
def get_movieids_(userid):
    return (ratings.loc[(ratings.user_id==userid),'movie_id'].tolist())

# Function to get the movie titles against the movie id.
def get_movie_title_(movieid):
    return (movies.loc[(movies.movie_id == movieid),'movie_title'].iloc[0])

def person_correlation_score(user1,user2):
    '''
    user1 & user2 : user ids of two users between which similarity score is to be calculated.
    '''
    both_watch_count = []
    for element in ratings.loc[ratings.user_id==user1,'movie_id'].tolist():
        if element in ratings.loc[ratings.user_id==user2,'movie_id'].tolist():
            both_watch_count.append(element)
    if len(both_watch_count) == 0 :
        return 0
    rating_sum_1 = sum([get_rating_(user1,element) for element in both_watch_count])
    rating_sum_2 = sum([get_rating_(user2,element) for element in both_watch_count])
    rating_squared_sum_1 = sum([pow(get_rating_(user1,element),2) for element in both_watch_count])
    rating_squared_sum_2 = sum([pow(get_rating_(user2,element),2) for element in both_watch_count])
    product_sum_rating = sum([get_rating_(user1,element) * get_rating_(user2,element) for element in both_watch_count])

    numerator = product_sum_rating - ((rating_sum_1 * rating_sum_2) / len(both_watch_count))
    denominator = sqrt((rating_squared_sum_1 - pow(rating_sum_1,2) / len(both_watch_count)) * (rating_squared_sum_2 - pow(rating_sum_2,2) / len(both_watch_count)))
    if denominator == 0:
        return 0
    return numerator/denominator

def get_recommendation_(userid):
    user_ids = ratings.user_id.unique().tolist()
    total = {}
    similariy_sum = {}

    # Iterating over subset of user ids.
    for user in user_ids[:100]:

        # not comparing the user to itself (obviously!)
        if user == userid:
            continue

        # Getting similarity score between the users.
        score = person_correlation_score(userid,user)

        # not considering users having zero or less similarity score.
        if score <= 0:
            continue

        # Getting weighted similarity score and sum of similarities between both the users.
        for movieid in get_movieids_(user):
            # Only considering not watched/rated movies
            if movieid not in get_movieids_(userid) or get_rating_(userid,movieid) == 0:
                total[movieid] = 0
                total[movieid] += get_rating_(user,movieid) * score
                similariy_sum[movieid] = 0
                similariy_sum[movieid] += score

    # Normalizing ratings
    ranking = [(tot/similariy_sum[movieid],movieid) for movieid,tot in total.items()]
    ranking.sort()
    ranking.reverse()

    # Getting movie titles against the movie ids.
    recommendations = [get_movie_title_(movieid) for score,movieid in ranking]
    return recommendations[:10]


## Simple POC Get a recomendation for an user, on new movies.
print(get_recommendation_(320))