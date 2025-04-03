import pandas as pd
import numpy as np


### Data Cleaning Methods

def load(json, chunk_size):
    """
    Load a large JSON into a dataframe, chunk_size rows at a time
    """
    review_chunks = []
    for chunk in pd.read_json(json, lines=True, chunksize=chunk_size):
        review_chunks.append(chunk)

    dataframe = pd.concat(review_chunks, ignore_index=True)

    return dataframe
    

def filter_business_data(business_dataset, chunk_size):
    '''
    Input Yelp business data and chunk_size, output all restaurants with >= 30 reviews
    '''
    # Read yelp business data into dataframe
    unfiltered_business_df = load(business_dataset, chunk_size)
    # Select all businesses with "Restaurants" category
    restaurants_df = unfiltered_business_df[(unfiltered_business_df["categories"].str.contains("restaurant", case=False, na=False))]
    # Select all restaurants with > 30 reviews
    reviewed_restaurants = restaurants_df[(restaurants_df["review_count"] >= 30)]

    return reviewed_restaurants


def filter_review_data(review_dataset, reviewed_restaurants, chunk_size):
    """
    Input Yelp review data and chunk size (number of reviews to read into memory at one time).
    Output all reviews about restaurants returned from filter_business_data
    """
    # Read yelp review data into dataframe
    unfiltered_reviews = load(review_dataset, chunk_size)
    # Select only reviews of businesses in our filtered dataset
    reviewed_restaurant_mask = unfiltered_reviews['business_id'].isin(reviewed_restaurants['business_id'])
    filtered_reviews = unfiltered_reviews[reviewed_restaurant_mask]

    return filtered_reviews


def filter_user_data(users_dataset, filtered_reviews, chunk_size):
    """
    Input Yelp user data, filtered review data, and chunk size (number of users to read into memory at a time).
    Output all users who have written reviews about restaurants returned from filter_business_data
    """
    # Read yelp user data into dataframe
    unfiltered_users = load(users_dataset, chunk_size)
    # Select only users who have written the reviews in our filtered dataset
    filtered_users = unfiltered_users[unfiltered_users['user_id'].isin(filtered_reviews['user_id'])]

    return filtered_users


def save(dataframe, file_path):
    """
    Save a dataframe as a JSON file. Input a dataframe and a location to save to
    """
    dataframe.to_json(str(file_path), orient='records', lines=True)
