import time

import argparse
parser = argparse.ArgumentParser()
    
# Definiere ein Argument
parser.add_argument('--variable', type=bool, help='Turn plusone on or off')

# Parse die Argumente
args = parser.parse_args()

# Zugriff auf die Variable
if str(args.variable) == "True":
    tomography_connections = True
elif str(args.variable) == "False":
    tomography_connections = False
else:
    print(str(args.variable))
    raise TypeError()

print(tomography_connections)
time.sleep(5)
print("End")