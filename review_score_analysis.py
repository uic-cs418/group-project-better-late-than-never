import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_average_review_score(restaurants_df, reviews_df):
    """
    Input filtered reviews and restaurants dataframes, return a dataframe with the average review score for each 
    restaurant in column 'avg_review_score' appended to restaurant dataframe
    """
    avg_review_scores = reviews_df.groupby('business_id')['stars'] \
                                        .mean() \
                                        .reset_index() \
                                        .rename(columns={'stars': 'avg_review_score'})

    merged_data = pd.merge(restaurants_df, avg_review_scores, on='business_id', how='left')

    return merged_data

