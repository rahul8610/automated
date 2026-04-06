import traceback

try:
    with open('trace.txt', 'r', encoding='utf-16le', errors='ignore') as f:
        print(f.read()[-500:])
except Exception as e:
    print(e)
