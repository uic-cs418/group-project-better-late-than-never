import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import data_cleaning
import review_score_analysis
import price_range_analysis


def make_restaurants_with_avg_stars(reviewsDF, restaurantsDF):
    
    tempReviews=reviewsDF.copy(deep=True)
    tempRestaurants=restaurantsDF.copy(deep=True)
    tempReviewsWithStars=review_score_analysis.calculate_average_review_score(tempReviews)
    restaurantsWithStars=tempReviewsWithStars.merge(tempRestaurants, on='business_id', how='inner')
    tempDF=restaurantsWithStars.drop(columns=["review_id","user_id", "text", "date", "stars_x", "useful", "funny", "cool", "stars_y"])
    return tempDF


def add_price_range_to_df(restaurants_df):
    tempDF=restaurants_df.copy(deep=True)
    tempDF=tempDF.dropna(subset="attributes")
    
    priceRangeList=[]
    
    for entry in tempDF["attributes"]:
        priceRangeList.append(entry.get("RestaurantsPriceRange2"))
    
    tempDF.insert(0, "PriceRange", priceRangeList)
    
    returnDF=tempDF.dropna(subset="PriceRange")
    returnDF=returnDF[returnDF["PriceRange"]!="None"]
    return returnDF

def show_price_plot(restaurants_df):
    axes=sns.boxplot(data=restaurants_df, x="PriceRange", y="avg_review_score", palette="mako")
    axes.set_xlabel("Price Range (as told by Yelp)")
    axes.set_ylabel("Average Review Score (1 to 5)")
    plt.title("Higher Prices Correlated with Higher Reviews")
    plt.show()
    return