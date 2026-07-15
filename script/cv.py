import pandas as pd

# استخدم شرطة مائلة للأمام أو raw string
df = pd.read_csv("C:/xampp/htdocs/crm/script/AmesHousing.csv")

print(df.info())
print(df.describe())
print(df.columns.tolist())