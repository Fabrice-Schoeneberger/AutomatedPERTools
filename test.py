
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--depths', type=int, nargs='+', help='Decide the depths of the pnt-samples. Default: [2,4,8,16]')
args = parser.parse_args()

print(args.depths)