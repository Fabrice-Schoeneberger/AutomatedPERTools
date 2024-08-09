import time

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--pntsamples', type=int, help='How many samples in PNT?')
parser.add_argument('--pntsinglesamples', type=int, help='How many single samples in PNT?')
parser.add_argument('--persamples', type=int, help='How many samples in PER?')

# Parse die Argumente
args = parser.parse_args()

print(args.test)
print("End")