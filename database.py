from peewee import *
from datetime import datetime
import os

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictpro.db')
db = SqliteDatabase(db_path)

class BaseModel(Model):
    class Meta:
        database = db

class WatchlistItem(BaseModel):
    ticker = CharField(unique=True)
    added_at = DateTimeField(default=datetime.now)

class PredictionHistory(BaseModel):
    ticker = CharField()
    date_run = DateTimeField(default=datetime.now)
    current_price = FloatField()
    predicted_price = FloatField()
    currency = CharField(default="$")
    ai_suggestion = CharField()
    model_used = CharField(default="XGBoost")

class ModelPerformance(BaseModel):
    ticker = CharField()
    model_name = CharField()
    r2_score = FloatField()
    rmse = FloatField()
    mae = FloatField()
    timestamp = DateTimeField(default=datetime.now)
    
def init_db():
    db.connect()
    db.create_tables([WatchlistItem, PredictionHistory, ModelPerformance], safe=True)
    db.close()
