import pandas as pd

# Flat Type Mapping
FLAT_TYPE_MAP = {
    "1 ROOM": 0,
    "2 ROOM": 1,
    "3 ROOM": 2,
    "4 ROOM": 3,
    "5 ROOM": 4,
    "EXECUTIVE": 5,
    "MULTI-GENERATION": 6,
}

# Clean Data
def clean_data(df):
    """
    Transforms raw HDB Resale DataFrame into cleaned format.

    Steps:
      1. Combine 'block' and 'street_name' into 'Address'.
      2. Extract lower bound of 'storey_range' into 'Storey'.
      3. Split 'month' into 'Year' and 'Month'.
      4. Convert 'remaining_lease' into fractional years.
      5. Map 'flat_type' to ordered integer categories.
      6. Select and rename relevant columns.

    Args:
        df: Raw DataFrame containing HDB resale data.

    Returns:
        Cleaned DataFrame with features:
        ['Year', 'Month', 'Town', 'Flat_Type', 'Address',
         'Storey', 'Floor_Area', 'Remaining_Lease', 'Price']
    """
    df_clean = df.copy()

    # Combine Block and Street Name to get Address
    df_clean["Address"] = df_clean["block"].astype(str) + " " + df_clean["street_name"].astype(str)
    
    # Take Lower Bound of Storey Range
    df_clean["Storey"] = pd.to_numeric(
        df_clean["storey_range"].str.extract(r"^(\d+)")[0], errors="coerce"
    )

    # Convert 'month' column to Year and Month
    df_clean['month_dt'] = pd.to_datetime(df_clean['month'], format='%Y-%m', errors='coerce')
    df_clean['Year'] = df_clean['month_dt'].dt.year
    df_clean['Month'] = df_clean['month_dt'].dt.month


    # Express Remaining Lease in Years
    df_clean["Remaining_Lease"] = df_clean["remaining_lease"].apply(
        lambda x: round(int(x.split()[0]) + (int(x.split()[2]) if "month" in x else 0)/12, 3)
        )
    
    # Convert Flat Type to Ordered Factor
    df_clean["Flat_Type"] = df_clean["flat_type"].map(FLAT_TYPE_MAP)
    
    # Select Relevant Columns
    selected = [
        "Year", "Month", "town", "Flat_Type", "Address",
        "Storey", "floor_area_sqm", "Remaining_Lease", "resale_price"
    ]
    df_clean = df_clean[selected].rename(columns={
        "town": "Town",
        "floor_area_sqm": "Floor_Area",
        "resale_price": "Price"
    })

    return df_clean

def remove_outliers(df, columns):
    """
    Calculates iqr range for specific columns

    Args:
        df: dataframe containing HDB resale data
        columns: specified column to calc iqr range

    Returns:
        Cleaned dataframe with iqr Q3 - Q1 yay
    """

    df_clean = df.copy()

    for col in columns:
        if col in df_clean.columns:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            # remove outliers
            lower_bound = q1 - 1.5 * iqr 
            upper_bound = q3 + 1.5 * iqr
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean
