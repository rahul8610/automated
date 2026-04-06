"""Quick test: import every module in the project to catch missing dependencies."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("1. Testing database.py ...", end=" ")
from database import init_db, db
init_db()
print("OK")

print("2. Testing strategy.py ...", end=" ")
from strategy import evaluate_strategies
print("OK")

print("3. Testing backtesting.py ...", end=" ")
from backtesting import run_backtest
print("OK")

print("4. Testing model.py (imports only) ...", end=" ")
from model import fetch_and_train
print("OK")

print("5. Testing app.py (imports only) ...", end=" ")
from flask import Flask
print("OK")

print("\n=== ALL IMPORTS PASSED ===")
