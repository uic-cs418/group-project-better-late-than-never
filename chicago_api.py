import requests
import pandas as pd
import time

def read_api_key(path="data/api_key_yelp.txt"):
    with open(path, "r") as file:
        return file.read().strip()

def read_neighborhoods(path="data/chicago_neighborhoods.txt"):
    neighborhoods = []
    with open(path, "r") as file:
        for line in file:
            neighborhoods.append(line.strip())
    return list(neighborhoods)

def location_search_params(api_key, location, **kwargs):
    url = "https://api.yelp.com/v3/businesses/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    url_params = {"location": f'{location}, Chicago, IL'}
    url_params = url_params | kwargs
    return url, headers, url_params


def api_get_request(url, headers, url_params):
    response = requests.get(url, headers=headers, params=url_params)
    return response.json()

def paginated_restaurants_search_requests(api_key, location, total):
    paginated_list = []
    for i in range(0, total, 40):
        url, headers, url_params = location_search_params(api_key, location, offset=i, limit=40, categories="restaurants", sort_by="rating")
        paginated_list.append((url, headers, url_params))
    return paginated_list

def get_chicago_restaurants_df(begin, end, api_key_path="data/api_key_yelp.txt", neighborhood_path="data/chicago_neighborhoods.txt"):
    # Prepare variables
    api_key = read_api_key(api_key_path)
    neighborhoods = read_neighborhoods(neighborhood_path)
    all_restaurants = pd.DataFrame(columns=["id", 
                                            "alias", 
                                            "name", 
                                            "image_url", 
                                            "is_closed", 
                                            "url", 
                                            "review_count", 
                                            "categories", 
                                            "rating", 
                                            "coordinates", 
                                            "transactions", 
                                            "price", 
                                            "location", 
                                            "phone", 
                                            "display_phone", 
                                            "distance", 
                                            "business_hours", 
                                            "attributes", 
                                            "neighborhood"]
                                            )
    
    for location in neighborhoods[begin:end]:
        # Make first API call to a neighborhood to get total # of hits
        url, headers, url_params = location_search_params(api_key, location, offset=0, limit=40, categories="restaurants")
        response = requests.get(url, headers=headers, params=url_params)
        time.sleep(0.5)
        if response.status_code == 200:
            # Paginate up to 240 of the top-rated restaurants for a given neighborhood
            data = response.json()
            restaurants = []
            print(f'{location} has {data["total"]} hits')
            total = min(240, data["total"])
            paginated_queries = paginated_restaurants_search_requests(api_key, location, total)
            for query in paginated_queries:
                response = api_get_request(*query)
                restaurants.extend(response.get("businesses", []))
                time.sleep(0.4)
            # Append neighborhood column to dataframe for each neighborhood (values with NA columns are dropped)
            restaurants_df = pd.DataFrame(restaurants)
            restaurants_df["neighborhood"] = location
            # Append neighborhood dataframe to the rest
            all_restaurants = pd.concat([all_restaurants, restaurants_df], ignore_index=True)
        
            
    # return pd.DataFrame(businesses)
    return all_restaurants