import time
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from haversine import haversine, Unit

# OneMap API Configuration
EMAIL = None # Enter your OneMaps API Email
PASSWORD = None # Enter your OneMaps API Password
TOKEN_URL = "https://www.onemap.gov.sg/api/auth/post/getToken"
ROUTE_URL = "https://www.onemap.gov.sg/api/public/routingsvc/route"

# Authenticate OneMap API
def authenticate():
    """
    Authenticate with OneMap API and returns access token
    """
    payload = {"email": EMAIL, "password": PASSWORD}
    try:
        response = requests.post(TOKEN_URL, json=payload)
        response.raise_for_status()
        return response.json().get("access_token", None)
    except requests.RequestException as e:
        print(f"Authentication failed: {e}")
        return None
    

# Nearest Location Finder
def nearest_loc(hdb_lat, hdb_lon, loc_df):
    """
    Find the nearest location (MRT, mall, school) to a given HDB coordinate.

    Args:
        hdb_lat: Latitude of HDB
        hdb_lon: Longitude of HDB
        loc_df: DataFrame of candidate locations with 'Lat', 'Long', 'Address'

    Returns:
        Tuple of nearest address, latitude, longitude, and whether it's within 1km
    """
    hdb_point = np.array([hdb_lat, hdb_lon])
    loc_points = loc_df[["Lat", "Long"]].to_numpy()
    dists = np.array([
        haversine(hdb_point, loc, unit=Unit.KILOMETERS) for loc in loc_points
    ])
    min_idx = np.argmin(dists)
    within_1km = None if dists[min_idx] == 0 else dists[min_idx] <= 1
    nearest = loc_df.iloc[min_idx]
    return nearest["Address"], nearest["Lat"], nearest["Long"], within_1km

    
# Obtain Distance between HDB and MRT/Mall/School from OneMap
def get_distance(hdb_lat, hdb_long, loc_lat, loc_long, api_token):
    """
    Get walking distance between two coordinates using OneMap routing service.

    Args:
        hdb_lat: Latitude of the starting HDB point.
        hdb_long: Longitude of the starting HDB point.
        loc_lat: Latitude of the destination point (e.g., MRT or mall).
        loc_long: Longitude of the destination point.
        api_token: OneMap API token for authentication.

    Returns:
        Total walking distance in meters if successful, otherwise None.
    """
    try:
        response = requests.get(
            ROUTE_URL,
            params={
                "start": f"{hdb_lat},{hdb_long}",
                "end": f"{loc_lat},{loc_long}",
                "routeType": "walk",
            },
            headers={"Authorization": api_token},
        )
        response.raise_for_status()
        return response.json().get("route_summary", {}).get("total_distance")
    
    except Exception:
        return None


# Engineer Distance Features (MRT, Mall, School)
def engineer_distance_features(hdbs, mrts, malls, schools, api_token):
    """
    Adds engineered features to HDBs: nearest MRT, Mall, School, and distances.

    Args:
        hdbs: DataFrame of HDBs with coordinates
        mrts: DataFrame of MRTs with coordinates
        malls: DataFrame of malls with coordinates
        schools: DataFrame of schools with coordinates
        apit_token: OneMaps API Token

    Returns:
        hdbs DataFrame with new features appended.
    """
    # Initialize Lists to Store Results
    nearest_mrt, mrt_distances = [], []
    nearest_mall, mall_distances = [], []
    nearest_school, within_1km = [], []

    for hdb_lat, hdb_long in tqdm(zip(hdbs["Lat"], hdbs["Long"]), total=len(hdbs)):
        mrt, mrt_lat, mrt_long, _ = nearest_loc(hdb_lat, hdb_long, mrts)
        mrt_dist = get_distance(hdb_lat, hdb_long, mrt_lat, mrt_long, api_token)
        mall, mall_lat, mall_long, _ = nearest_loc(hdb_lat, hdb_long, malls)
        mall_dist = get_distance(hdb_lat, hdb_long, mall_lat, mall_long, api_token)
        school, _, _, school_1km = nearest_loc(hdb_lat, hdb_long, schools)

        nearest_mrt.append(mrt)
        mrt_distances.append(mrt_dist)
        nearest_mall.append(mall)
        mall_distances.append(mall_dist)
        nearest_school.append(school)
        within_1km.append(school_1km)
    
    # Add New Columns to HDBs Dataframe
    hdbs["Nearest_MRT"] = nearest_mrt
    hdbs["Distance_MRT"] = mrt_distances
    hdbs["Nearest_Mall"] = nearest_mall
    hdbs["Distance_Mall"] = mall_distances
    hdbs["Nearest_Pri_Sch"] = nearest_school
    hdbs["Within_1km_of_Pri"] = within_1km

    return hdbs

if __name__ == "__main__":
    # Authenticate and get API Token
    api_token = authenticate()
    if api_token is None:
        print("Authentication failed. Exiting...")
        exit(1)

    # Read CSVs
    mrts = pd.read_csv("datasets/coordinates/MRT_LatLong.csv")
    malls = pd.read_csv("datasets/coordinates/Mall_LatLong.csv")
    schools = pd.read_csv("datasets/coordinates/School_LatLong.csv")
    hdbs = pd.read_csv("datasets/coordinates/HDB_LatLong.csv")

    # Get HDB Distance Features in Batches of 100 (30s cooldown for API ratelimit)
    BATCH_SIZE = 100
    BATCHES = (len(hdbs) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(BATCHES):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(hdbs))
        batch = hdbs.iloc[start:end].copy()
        batch_engineered_features = engineer_distance_features(batch, mrts, malls, schools, api_token)
        batch_engineered_features.to_csv("datasets/HDB_Features.csv",
                                         index=False, mode='a', header=False)
        time.sleep(30)