import pandas as pd
from pathlib import Path
from data_cleaning import clean_data

"""
Pipeline is to be used when there are updates to Raw Data, RPI, or HDB Features
"""

# Default File Paths
RAWDATA_CSV = Path("datasets/Raw_ResaleData.csv")
CLEANDATA_CSV = Path("datasets/Cleaned_ResaleData.csv")
FINALDATA_CSV = Path("datasets/Final_ResaleData.csv")
HDB_FEATURES_CSV = Path("datasets/HDB_Features.csv")

MATURE_ESTATES = [
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH", "CENTRAL", "CLEMENTI",
    "GEYLANG", "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "QUEENSTOWN", "SERANGOON",
    "TAMPINES", "TOA PAYOH"
]

if __name__ == "__main__":
    # Convert Raw Data to Clean Data
    df_raw = pd.read_csv(RAWDATA_CSV)
    df_clean = clean_data(df_raw)
    df_clean.to_csv(CLEANDATA_CSV, index=False)

    # Add Engineered Features (Distances, RPI) to obtain Final Dataset
    hdbs = pd.read_csv(HDB_FEATURES_CSV)
    final_df = pd.merge(df_clean, hdbs[["Address", "Distance_MRT", "Distance_Mall", "Within_1km_of_Pri"]], on='Address', how='left')

    # Classify Towns into Mature/Non-Mature Estate
    final_df["Mature"] = final_df["Town"].isin(MATURE_ESTATES)
    
    final_df.to_csv(FINALDATA_CSV, index=False)
