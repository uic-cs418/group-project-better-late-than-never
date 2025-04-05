import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def addAverageStarsColumn(restaurants_df, reviews_df):


    """
    Inputs: restaurants_df, a dataframe where rows represent a restaurant with the column "business_id".
            reviews_df, a dataframe with the columns "business_id" and "stars".
            
    Outputs: A dataframe with all the same rows and columns as restaurants_df, but with the added column of "average_stars", representing
            the average review score for that restaurant.
    """
    
    businessStars=reviews_df[["business_id", "stars"]]
    groupedReviews=businessStars.groupby("business_id").mean()
    groupedReviewsDF=groupedReviews.reset_index()
    groupedReviewsDF=groupedReviewsDF.rename(columns={"business_id": "business_id", "stars" : "average_stars"})
    groupedReviewsDF.head()
    
    restaurantsWithAverageStars=pd.merge(restaurants_df, groupedReviewsDF, how="inner", on="business_id")
    
    return restaurantsWithAverageStars



def addPriceRangeColumn(restaurants_df):

    """
    Inputs: restaurants_df, a dataframe where rows represent a restaurant with the column "attributes".
            
    Outputs: A dataframe with all the same rows and columns as restaurants_df, but with the added column of "PriceRange", representing
            the reported expected cost of eating at the restaurant by Yelp reviewers (graded from 1 to 4, where 1 is least expensive).
    """
    droppedAttributes=restaurants_df.copy(deep=True)
    droppedAttributes=droppedAttributes.dropna(subset="attributes")
    priceRangeList=[]

    for entry in droppedAttributes["attributes"]:
        priceRangeList.append(entry.get("RestaurantsPriceRange2"))
    restaurantsWithPriceRanges=droppedAttributes.dropna(subset="PriceRange")
    restaurantsWithPriceRanges=restaurantsWithPriceRanges[restaurantsWithPriceRanges["PriceRange"]!="None"]
    
    
    return restaurantsWithPriceRanges