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
    confidence_score = FloatField(null=True)
    strategy_explanation = TextField(null=True)
    model_used = CharField(default="XGBoost")

class ModelPerformance(BaseModel):
    ticker = CharField()
    model_name = CharField()
    r2_score = FloatField()
    rmse = FloatField()
    mae = FloatField()
    timestamp = DateTimeField(default=datetime.now)
    
class BacktestSummary(BaseModel):
    ticker = CharField()
    initial_capital = FloatField()
    final_capital = FloatField()
    total_pnl = FloatField()
    win_rate = FloatField()
    total_trades = IntegerField()
    max_drawdown = FloatField()
    timestamp = DateTimeField(default=datetime.now)

def init_db():
    db.connect(reuse_if_open=True)
    db.create_tables([WatchlistItem, PredictionHistory, ModelPerformance, BacktestSummary], safe=True)
    db.close()
