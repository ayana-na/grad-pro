import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("="*60)
print(" new one")
print("="*60)


engine = create_engine("mysql+pymysql://root:@127.0.0.1/real_estate_ai", echo=False)

try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info(" connection done")
except Exception as e:
    logger.error(f" err: {e}")
    exit(1)

with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS properties"))
    conn.execute(text("TRUNCATE TABLE leads"))
    conn.execute(text("TRUNCATE TABLE lead_recommendations"))
    logger.info("clean done")

csv_path = "AmesHousing.csv"

if not os.path.exists(csv_path):
    logger.error(f" file{csv_path} not exist!")
    exit(1)

df = pd.read_csv(csv_path)
logger.info(f" load {len(df)} property from Ames Housing")


properties_df = pd.DataFrame()

properties_df['price'] = df['SalePrice']
properties_df['bedrooms'] = df['Bedroom AbvGr'].fillna(0).astype(int)
properties_df['bathrooms'] = df['Full Bath'].fillna(0) + 0.5 * df['Half Bath'].fillna(0)
properties_df['sqft'] = (
    df['1st Flr SF'].fillna(0) +
    df['2nd Flr SF'].fillna(0) +
    df['Total Bsmt SF'].fillna(0)
).clip(lower=300)

properties_df['location_lat'] = 42.034722
properties_df['location_long'] = -93.620000

properties_df['overall_qual'] = df['Overall Qual']
properties_df['overall_cond'] = df['Overall Cond']
properties_df['year_built'] = df['Year Built'].fillna(1990).astype(int)
properties_df['year_remodel'] = df['Year Remod/Add'].fillna(df['Year Built'])
properties_df['bsmt_sf'] = df['Total Bsmt SF'].fillna(0)
properties_df['deck_sf'] = df['Wood Deck SF'].fillna(0)
properties_df['porch_sf'] = df['Open Porch SF'].fillna(0)
properties_df['fireplaces'] = df['Fireplaces'].fillna(0)
properties_df['garage_cars'] = df['Garage Cars'].fillna(0)
properties_df['garage_area'] = df['Garage Area'].fillna(0)


categorical_cols = ['MS Zoning', 'Street', 'Alley', 'Lot Shape', 'Land Contour', 'Utilities', 
                    'Lot Config', 'Land Slope', 'Neighborhood', 'Condition 1', 'Condition 2', 
                    'Bldg Type', 'House Style', 'Roof Style', 'Roof Matl', 'Exterior 1st', 
                    'Exterior 2nd', 'Mas Vnr Type', 'Exter Qual', 'Exter Cond', 'Foundation', 
                    'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2', 
                    'Heating', 'Heating QC', 'Central Air', 'Electrical', 'Kitchen Qual', 
                    'Functional', 'Fireplace Qu', 'Garage Type', 'Garage Finish', 'Garage Qual', 
                    'Garage Cond', 'Paved Drive', 'Pool QC', 'Fence', 'Misc Feature', 
                    'Sale Type', 'Sale Condition']

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None').astype(str)
        le = LabelEncoder()
        col_name = col.lower().replace(' ', '_')
        properties_df[col_name] = le.fit_transform(df[col])


np.random.seed(42)
properties_df['listing_date'] = datetime.now() - pd.to_timedelta(
    np.random.randint(1, 730, size=len(properties_df)), unit='D'
)


properties_df = properties_df[properties_df['price'] > 0].drop_duplicates()


properties_df.to_sql(
    name='properties',
    con=engine,
    if_exists='replace',
    index=True,                    
    index_label='property_id',
    method='multi',
    chunksize=500
)

logger.info(f" new insert {len(properties_df)} done")


with engine.connect() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM properties")).scalar()
    logger.info(f" properties count now: {count}")

print("\n" + "="*60)
print(" load done Ames Housing ")
print("="*60)