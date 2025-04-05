import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def averageRatingbyUsers(users_df, reviews_df):
    """
    histogram of average ratings given by users
    """
    avgStarUser = reviews_df.groupby("user_id")["stars"].mean().reset_index()
    avgStarUser.columns = ["user_id", "avgRatingGiven"]
    
    possiblebias = users_df.merge(avgStarUser, on="user_id", how="inner")

    plt.figure(figsize=(10, 6))
    sns.histplot(possiblebias["avgRatingGiven"], bins=20)
    plt.title("distribution of average ratings given by users")
    plt.xlabel("average rating")
    plt.ylabel("# of Users")
    plt.grid(True)
    plt.show()


def numReviewDistributionbyUsers(users_df):
    """
    groups users by number of reviews written
    """
    bins = [0, 2, 5, 10, 20, 50, 100, 500, float("inf")]
    labels = ["1-2","3-5","6-10","11-20","21-50","51-100","101-500","500+"]

    users_df["review_category"] = pd.cut(
        users_df["review_count"], bins=bins, labels=labels,
        right=False, include_lowest=True
    )
    review_counts = users_df["review_category"].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=review_counts.index, y=review_counts.values)
    plt.xlabel("# of review ranges")
    plt.ylabel("# of users")
    plt.title("# of users grouped by # of reviews")
    plt.grid(axis="y", linestyle="--")
    plt.show()
