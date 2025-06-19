from pymongo.mongo_client import MongoClient
import pandas as pd
import json

# MongoDB Atlas URI
uri = "mongodb+srv://Sourav:Sourav0987@cluster0.wwq8ftz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create MongoDB client using Atlas URI
client = MongoClient(uri)

# Define DB and Collection
DATABASE_NAME = "sensors"
COLLECTION_NAME = "waferfaults"
collection = client[DATABASE_NAME][COLLECTION_NAME]

# Read and clean CSV
df = pd.read_csv("wafer_23012020_041211.csv")
df = df.drop(columns=["Unnamed: 0"], errors='ignore')


json_str = df.to_json(orient="records")    
json_record = json.loads(json_str)         

# Optional: print type check
print(type(json_record))

# Insert into MongoDB
collection.insert_many(json_record)

