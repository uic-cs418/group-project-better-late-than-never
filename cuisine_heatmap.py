"""
Generate a heatmap of mean and median average review scores by cuisine.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data_cleaning

# Mapping raw category tags into cuisine buckets
CUISINE_MAP = {
    'American': [
        'American (Traditional)', 'American (New)', 'Breakfast & Brunch',
        'Sandwiches', 'Burgers', 'Fast Food', 'Coffee & Tea', 'Cafes'
    ],
    'Mexican': ['Mexican', 'Tacos', 'Tex-Mex'],
    'Italian': ['Italian', 'Pizza'],
    'Chinese': ['Chinese', 'Dim Sum'],
    'Japanese': ['Japanese', 'Sushi Bars'],
    'Seafood': ['Seafood'],
    'Mediterranean': ['Mediterranean', 'Greek', 'Turkish'],
    'Barbecue': ['Barbeque'],
    'Indian': ['Indian'],
    'Thai': ['Thai'],
    'Vietnamese': ['Vietnamese']
}

def map_cuisine(cat_string):
    """
    Assign a cuisine bucket based on mapping keywords.
    Defaults to 'Other' if no match.
 """
    if not isinstance(cat_string, str):
        return 'Other'
    for cuisine, keywords in CUISINE_MAP.items():
        for kw in keywords:
            if kw.lower() in cat_string.lower():
                return cuisine
    return 'Other'


def heatmap():
    #Load filtered restaurants and reviews
    chunk_size = 100_000
    biz_df = data_cleaning.filter_business_data(
        "data/yelp_academic_dataset_business.json", chunk_size
    )
    rev_df = data_cleaning.filter_review_data(
        "data/yelp_academic_dataset_review.json", biz_df, chunk_size
    )

    # Compute average review score per restaurant
    avg_scores = (
        rev_df.groupby('business_id')['stars']
        .mean()
        .reset_index(name='avg_review_score')
    )
    df = pd.merge(biz_df, avg_scores, on='business_id', how='left')

    #  Map cuisines
    df['food_category'] = df['categories'].apply(map_cuisine)

    # Aggregate mean and median of avg_review_score by cuisine
    cuisine_stats = (
        df.groupby('food_category')['avg_review_score']
          .agg(['mean','median'])
          .sort_values('mean', ascending=False)
    )

    #Plot heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(
        cuisine_stats,
        annot=True,
        fmt=".2f",
        cmap='YlGnBu',
        cbar_kws={'label': 'Avg Review Score'}
    )
    plt.title('Mean & Median Avg Review Score by Cuisine')
    plt.ylabel('Cuisine')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
