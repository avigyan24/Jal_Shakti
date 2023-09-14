from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pymongo import MongoClient
import json
from bson import ObjectId
from fastapi.responses import JSONResponse
from bson import ObjectId
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS



app = FastAPI()

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
uri = "mongodb+srv://admin:test1234@cluster0.ifzbffj.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# MongoDB configuration
MONGODB_URL = "mongodb+srv://admin:test1234@cluster0.ifzbffj.mongodb.net/?retryWrites=true&w=majority"
DATABASE_NAME = "Cluster0"
COLLECTION_NAME = "predictions"



client=MongoClient("mongodb+srv://admin:test1234@cluster0.ifzbffj.mongodb.net/?retryWrites=true&w=majority")
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)
# Function to extract GPS data from Exif metadata
def get_geolocation_info(exif_data):
    geolocation_info = {}
    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)
        if tag_name == "GPSInfo":
            for subtag, subvalue in value.items():
                subtag_name = GPSTAGS.get(subtag, subtag)
                geolocation_info[subtag_name] = subvalue
    return geolocation_info



origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("classifier_model2.h5")

CLASS_NAMES = ["Urban Flooding", "Drainage", "Polluted lake"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
# Set the custom JSON encoder
app.json_encoder = CustomJSONEncoder

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    # Store the prediction in MongoDB
    prediction_data = {
        'class': predicted_class,
        'confidence': confidence
    }
    collection.insert_one(prediction_data)



    # Convert ObjectId to string in the prediction dictionary
    prediction_data['_id'] = str(prediction_data['_id'])

    # Return the prediction dictionary as a JSON response
    return JSONResponse(content=prediction_data)

@app.on_event("shutdown")
async def shutdown_event():
    client.close()

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)