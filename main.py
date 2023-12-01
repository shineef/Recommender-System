import zipfile
import csv
from collections import defaultdict
import scipy
import scipy.optimize
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from deep_model import train_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import csv
from datetime import datetime

current_hour = datetime.now().hour

paths = [
    "X:/IR/Project/Books.csv.zip",
    "X:/IR/Project/Ratings.csv.zip",
    "X:/IR/Project/Users.csv.zip"
]

books = {}
ratings = defaultdict(list)
users = {}
similarities = []

similarity_measure = 'cosine'

# Loop over each path
for path in paths:
    # Open and extract each zip file
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall("X:/IR/Project/IR/Project/")
        csv_path = zip_ref.namelist()[0]

    # Open the extracted CSV file
    with open(f"X:/IR/Project/IR/Project/{csv_path}", 'r') as file:
        reader = csv.reader(file)
        # Get the first row of the CSV file (usually contains the attributes)
        attributes = next(reader)

        # Process each line in the CSV file
        for line in file:
            fields = line.strip().split(',')
            record = dict(zip(attributes, fields))

            # Add the record to the appropriate dictionary
            if 'Books.csv' in csv_path:
                books[record['ISBN']] = record
            elif 'Ratings.csv' in csv_path:
                ratings[record['ISBN']].append(record)
            elif 'Users.csv' in csv_path:
                users[record['User-ID']] = record

user_encoder = LabelEncoder()
book_encoder = LabelEncoder()

# Flatten the ratings dictionary into a list of dictionaries
ratings_list = [dict(ISBN=isbn, **{k: v for k, v in rating.items() if k != 'ISBN'}) for isbn, ratings in ratings.items() for rating in ratings]

# Convert the list into a DataFrame
ratings_df = pd.DataFrame(ratings_list)

# Convert the 'User-ID' and 'ISBN' columns to numeric
ratings_df['User-ID'] = user_encoder.fit_transform(ratings_df['User-ID'])
ratings_df['ISBN'] = book_encoder.fit_transform(ratings_df['ISBN'])

# Drop any rows with NaN values
ratings_df = ratings_df.dropna()

# Calculate the number of unique users and books
num_users = ratings_df['User-ID'].nunique()
num_books = ratings_df['ISBN'].nunique()

# Train the model
# model = train_model(ratings_df, num_users, num_books)
model = load_model('book_recommender_model.h5')

# Create dictionaries for user and book encodings
user2user_encoded = {x: i for x, i in zip(user_encoder.classes_, user_encoder.transform(user_encoder.classes_))}
book2book_encoded = {x: i for x, i in zip(book_encoder.classes_, book_encoder.transform(book_encoder.classes_))}

def cosine_similarity(isbn1, isbn2):
    # Get the ratings for each book
    ratings1 = ratings[isbn1]
    ratings2 = ratings[isbn2]

    # Get the user IDs and ratings of the users who rated each book
    user_ratings1 = {r['User-ID']: r['Book-Rating'] for r in ratings1}
    user_ratings2 = {r['User-ID']: r['Book-Rating'] for r in ratings2}

    # Find common raters
    common_raters = [u for u in user_ratings1.keys() if u in user_ratings2.keys()]

    # If there are no common raters, return 0
    if not common_raters:
        return 0

    # Get the ratings from the common raters
    ratings1 = [user_ratings1[u] for u in common_raters]
    ratings2 = [user_ratings2[u] for u in common_raters]

    # Calculate and return the cosine similarity
    return sklearn_cosine_similarity([ratings1], [ratings2])[0][0]

# Now, let's implement item-based collaborative filtering
def item_similarity(isbn1, isbn2):
    # Get the ratings for each book
    ratings1 = ratings[isbn1]
    ratings2 = ratings[isbn2]

    # Get the user IDs and ratings of the users who rated each book
    user_ratings1 = {r['User-ID']: r['Book-Rating'] for r in ratings1}
    user_ratings2 = {r['User-ID']: r['Book-Rating'] for r in ratings2}

    # Find common raters
    common_raters = [u for u in user_ratings1.keys() if u in user_ratings2.keys()]

    # If there are less than two common raters, return 0
    if len(common_raters) < 2:
        return 0

    # Get the ratings from the common raters
    ratings1 = [user_ratings1[u] for u in common_raters]
    ratings2 = [user_ratings2[u] for u in common_raters]

    # Calculate and return the Pearson correlation
    return pearsonr(ratings1, ratings2)[0]

def recommend_books(user_id, similarity_measure=similarity_measure, num_recommendations=5):
    global similarities
    books = {}
    with open('Books.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            isbn, title, author, publication_year, publisher, *_ = row
            books[isbn] = {
                'Book-Title': title,
                'Book-Author': author,
                'Year-Of-Publication': publication_year,
                'Publisher': publisher,
                'ISBN': isbn
            }

    # Get the books this user has rated
    user_ratings = [r for rating_list in ratings.values() for r in rating_list if r['User-ID'] == user_id]

    if user_ratings:
        # Calculate the similarity between each of these books and all other books
        if similarity_measure == 'pearson':
            similarities = [(isbn, item_similarity(r['ISBN'], isbn)) for r in user_ratings for isbn in books.keys()]
        elif similarity_measure == 'cosine':
            similarities = [(isbn, cosine_similarity(r['ISBN'], isbn)) for r in user_ratings for isbn in books.keys()]
        elif similarity_measure == 'deep':
            # Get the books the user hasn't rated yet
            user_ratings_isbn = [r['ISBN'] for r in user_ratings]
            # Filter the unrated books
            unrated_books = {isbn: book for isbn, book in books.items() if isbn not in user_ratings_isbn and isbn in book2book_encoded}

            # Predict the user's ratings for the unrated books
            user_arr = np.array([user2user_encoded[user_id]] * len(unrated_books))
            book_arr = np.array([book2book_encoded[isbn] for isbn in unrated_books.keys()])
            predictions = model.predict([user_arr, book_arr])

            # Get the indices of the top predictions
            top_indices = predictions.flatten().argsort()[-num_recommendations:][::-1]

            # Get the recommended books
            recommended_books = {isbn: unrated_books[isbn] for i, isbn in enumerate(unrated_books) if i in top_indices}

            return [(books[isbn], "Based on your ratings") for isbn in recommended_books]
        
        # Based on the predicted ratings, recommend the top books
        # recommended_books = [(isbn, predict_rating(user_id, isbn, similarities)) for isbn in books.keys()]
        # recommended_books.sort(key=lambda x: x[1], reverse=True)
        # return [(books[isbn], "Based on your ratings", rating) for isbn, rating in recommended_books[:num_recommendations]]

        # Based on the similarities, recommend the top books
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_books = similarities[:num_recommendations]
        return [(books[isbn], "Based on your ratings") for isbn, _ in top_books]
    else:
        # If the user hasn't rated any books, recommend the most popular books
        popular_books = sorted(ratings.items(), key=lambda x: len(x[1]), reverse=True)
        return [(books[isbn], "Popular book") for isbn, _ in popular_books[:num_recommendations]]

def predict_rating(user_id, isbn, similarities):
    # Get the books the user has rated
    user_ratings = [r for rating_list in ratings.values() for r in rating_list if r['User-ID'] == user_id]

    # Get the similarities for the given book
    book_similarities = [similarity for book_isbn, similarity in similarities if book_isbn == isbn]

    # Calculate the weighted average of the user's ratings for the most similar books
    weighted_ratings = [float(r['Book-Rating']) * similarity for r, similarity in zip(user_ratings, book_similarities)]
    prediction = sum(weighted_ratings) / sum(book_similarities)

    return prediction

def add_rating(user_id, location, age, isbn, book_rating):
    # Add user to Users.csv
    with open('Users.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, location, age])

    # Add rating to Ratings.csv
    with open('Ratings.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, isbn, book_rating])

def delete_user_info(user_id):
    # Delete from 'Users.csv'
    with open('Users.csv', 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row[0] != user_id]

    with open('Users.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # Delete from 'Ratings.csv'
    with open('Ratings.csv', 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row[0] != user_id]

    with open('Ratings.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Deleted info for User {user_id}")

# Test the recommendation system
# print(recommend_books('1'))
test_data = list(zip(ratings_df['User-ID'], ratings_df['ISBN'], ratings_df['Book-Rating']))

if current_hour < 12:
    part_of_day = 'morning'
elif 12 <= current_hour < 18:
    part_of_day = 'afternoon'
else:
    part_of_day = 'evening'

user_id = input("Enter User-ID: ")

if user_id == '0':
    print(f"Good {part_of_day}, Xinyi")
else:
    print(f"Good {part_of_day}, User {user_id}")

while True:
    if user_id == '0':
        mode = input("What do you want to do? (rate/search/delete/modify/exit): ")
    else:
        mode = input("What do you want to do? (rate/search/delete/exit): ")

    if mode.lower() == 'exit':
        break

    elif mode.lower() == 'rate':
        # Rating mode
        location = input("Enter Location: ")
        age = input("Enter Age: ")
        isbn = input("Enter ISBN: ")
        book_rating = input("Enter Book-Rating: ")

        add_rating(user_id, location, age, isbn, book_rating)

    elif mode.lower() == 'search':
        # Search mode
        recommended_books = recommend_books(user_id)

        for book, reason in recommended_books:
            print(f"Title: {book['Book-Title']}")
            print(f"Author: {book['Book-Author']}")
            print(f"Publication Year: {book['Year-Of-Publication']}")
            print(f"Publisher: {book['Publisher']}")
            print(f"ISBN: {book['ISBN']}")
            print(f"Reason: {reason}")
            print("-----")

    elif mode.lower() == 'delete':
        # Delete mode
        delete_user_info(user_id)

    elif user_id == '0' and mode.lower() == 'modify':
        # Modify mode
        new_similarity_measure = input("Enter new similarity measure: ")
        similarity_measure = new_similarity_measure
        print(f"Similarity measure changed to {similarity_measure}")

