import os
import pickle

# Define the name of the directory
directory = "new_folder"

# Create the directory
os.makedirs(directory, exist_ok=True)
directory += "/"
a = [1,2,3]
with open(directory+"noisedataframe.pickle", "wb") as f:
    noisedataframe = pickle.dump(a, f)
