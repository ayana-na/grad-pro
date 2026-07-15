import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib

engine = create_engine("mysql+pymysql://root:@127.0.0.1/real_estate_ai")

query = """
SELECT
price,
market_estimated_price,
sqft,
bedrooms
FROM properties
"""

df = pd.read_sql(query,engine)

X = df[[
"sqft",
"bedrooms"
]]

y = df["market_estimated_price"]

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=42
)

model = GradientBoostingRegressor(
n_estimators=200,
learning_rate=0.05,
max_depth=3
)

model.fit(X_train,y_train)

joblib.dump(model,"deal_model.pkl")

print("Deal model trained!")