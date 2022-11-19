# from pandas.io.json import json_normalize

from pandas import json_normalize
from pymongo import MongoClient
import pymongo
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns

from bson.objectid import ObjectId
from bson.decimal128 import Decimal128
import json


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId) or isinstance(o, Decimal128):
            return (str(o))
        return json.JSONEncoder.default(self, o)


local_uri = "mongodb://localhost:27017"
local_client = pymongo.MongoClient(local_uri)

weather_db = local_client["100YWeatherSmall"]["data"]

weather_filter = {
    "$match":{
        "dewPoint.value" : { "$lt" : 900.0 },
        "pressure.value" : { "$lt" : 9000.0 },
        "airTemperature.value" : { "$lt" : 900.0 } }
}

weather_projection = {
    "$project":{
        "_id":0,
        "airTemperature": "$airTemperature.value",
        "dewPoint": "$dewPoint.value",
        "pressure": "$pressure.value"
    }
}

weather_projection = {
    "$project":{
        "_id":0,
        "airTemperature.value": 1,
        "dewPoint.value": 1,
        "pressure.value": 1
    }
}

sample_stage = {"$sample":{"size":10000}}

cursor = weather_db.aggregate([
    weather_filter,
    weather_projection,
    sample_stage
])

weather_data = list(cursor)

print(weather_data[0])
df = json_normalize(weather_data)
df.head
# %matplotlib inline
sns.pairplot(df)


df_x = df.drop(['airTemperature.value'], axis=1)
df_y = df['airTemperature.value']

reg = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)

reg.fit(x_train, y_train)
reg.coef_
reg.intercept_
reg.predict(x_test)
print(np.mean((reg.predict(x_test) - y_test)**2))
