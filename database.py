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
    ai_suggestion = CharField()
    
def init_db():
    db.connect()
    db.create_tables([WatchlistItem, PredictionHistory], safe=True)
    db.close()
