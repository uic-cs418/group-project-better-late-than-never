import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_average_review_score(reviews_df):
    """
    Input filtered reviews dataframe, return a dataframe with the average review score for each 
    restaurant in column 'avg_review_score' appened to reviews dataframe
    """
    avg_review_scores = reviews_df.groupby('business_id')['stars'] \
                                        .mean() \
                                        .reset_index() \
                                        .rename(columns={'stars': 'avg_review_score'})

    merged_data = pd.merge(reviews_df, avg_review_scores, on='business_id', how='left')

    return merged_data

def calculate_average_score(df, group_key='business_id', score_col='stars', new_col='avg_score'):
    """
    Calculate the average score per group.
    """
    avg_scores = df.groupby(group_key)[score_col] \
                   .mean() \
                   .reset_index() \
                   .rename(columns={score_col: new_col})
    return avg_scores

def merge_data(left_df, right_df, key='business_id', how='left'):
    """
    Merge two DataFrames on a given key.
    """
    merged_df = pd.merge(left_df, right_df, on=key, how=how)
    return merged_df

def plot_review_distribution(reviews_df):
    """
    Plots a histogram of individual review scores.
    """
    plt.figure(figsize=(8,6))
    plt.hist(reviews_df['stars'], bins=range(1, 7), align='left', edgecolor='black')
    plt.title("Distribution of Individual Review Scores")
    plt.xlabel("Review Score")
    plt.ylabel("Frequency")
    plt.xticks(range(1, 6))
    plt.show()

def plot_average_score_distribution(avg_scores_df):
    """
    Plots a histogram of average review scores per restaurant.
    """
    plt.figure(figsize=(8,6))
    plt.hist(avg_scores_df['avg_review_score'], bins=20, edgecolor='black')
    plt.title("Distribution of Average Review Scores per Restaurant")
    plt.xlabel("Average Review Score")
    plt.ylabel("Number of Restaurants")
    plt.show()

