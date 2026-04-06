import model
import traceback
print("Starting script")
try:
    res, err = model.fetch_and_train('RELIANCE.NS')
    if err:
        print("ERROR: ", err)
except Exception as e:
    traceback.print_exc()
