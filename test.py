import time

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--setqubits', type=int, nargs='+', help='Which qubits to use?', default=[1,2,3,4])
parser.add_argument('--sum', type=str, help='Turn sumation on or off! Default: off')
parser.add_argument('--isset', '-s', help='Turn sumation on or off! Default: off', default=False, action='store_true')

# Parse die Argumente
args = parser.parse_args()

for arg_name, arg_value in vars(args).items():
    print(arg_name, arg_value)

#print("%s ist ein satz mit %s"% ("Hallo welt", "welt"))
print("End")