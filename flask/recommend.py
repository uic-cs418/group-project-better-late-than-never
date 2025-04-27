import pandas as pd


def get_best_restaurants(df, category, neighborhood, price_range):
    """
    Input:

    dataframe "df" with columns for business name (name), categories the business falls under (categories), the neighborhood that restaurant is in (neighborhood), the average review scores (average_stars), and the price range of that restaurant on a scale from 1 to 4, 1 being least expensive (PriceRange).

    string "category" to be searched for in the categories that the business falls under.

    string "neighborhood" to be filtered by in the neighborhood that restaurant is located in.

    int [1,4] "price_range" to be filtered by in the price_range of the restaurants.

    Output:

    dataframe with (at most) 5 rows, taken from the highest reviewed restaurants that fall under the appropriate conditions.
    """

    # make a temp DF so none of our filtering is reflected in the original.
    tempDF = df.copy(deep=True)
    # select all rows with the appropriate category.
    # tempDF = tempDF[(tempDF["categories"].str.contains(category, case=False, na=False))]
    # print(tempDF.head())
    # select all restaurants that are in the correct price range.
    # tempDF["PriceRange"] = tempDF.price.astype("int")
    # tempDF = tempDF[(tempDF["price"] == price_range)]
    tempDF = tempDF[(tempDF["price"].str.contains(price_range, case=False, na=False))]
    # select all restaurants in the appropriate neighborhood.
    tempDF = tempDF[
        (tempDF["neighborhood"].str.contains(neighborhood, case=False, na=False))
    ]
    # sort the resulting DF by average review score
    newDF = tempDF.sort_values(by=["rating"], inplace=False, ascending=False)
    # select the top 5 results. Note that bc of the slice semantics, it's possible for fewer than 5 rows to be gotten here, and no error will be thrown.
    newDF = newDF.iloc[:5]

    return newDF["name"].tolist()
