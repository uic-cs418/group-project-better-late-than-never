# Yelp Review Aggregator 
### By Team Better Late than Never

### Overview  
This project analyzes Yelp business, user, and review data to identify patterns in restaurant reviews. We explore:  
- Which city has the best food?  
- How do we know who makes 'good' food? 
- We want restaurant reccomendations.

## Data  
We are using the **Yelp Academic Dataset**, which includes:  
- `yelp_academic_dataset_business.json` 
- `yelp_academic_dataset_user.json`  
- `yelp_academic_dataset_review.json`

## Cleaning data:
- Remove non-restaurants
- Remove restaurants with < 30 reviews
- Remove reviews on those removed businesses
- Remove users who did not review restaurants in this set

## EDA Questions:
- Distribution of review scores and reviews
- Bias in reviews (e.g. not centered on 3 out of 5 stars)
- Correlations with average review score

## Machine Learning Models
- ML #1: Binary Review Classification (‘Bad’ vs. ‘Good’  Review Text)
- ML #2: 3-Class Review Classification (‘Bad’, ‘Good’ and ‘Great’ Review Text)

## Takeaways
- Yelp data is biased
- Bias in Yelp data is user input, at least in part

## Solution
- App Demo
- Used Yelp API to pull restaurants from Chicago neighborhoods
- https://yelp-recs.crowsnet.io
