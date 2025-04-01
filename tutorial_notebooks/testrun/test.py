import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--noise_strengths', type=float, nargs='+', help='Decide the depths of the pnt-samples. Default: [2,4,8,16]')
parser.add_argument('--persamples', type=int, help='How many samples in PER? Default: 100', default=100)
parser.add_argument('--expectations', type=str, nargs='+', help='Decide the expectation values which whill be measured at PER. Default: Z on all used qubits')
args = parser.parse_args()
persamples = args.persamples
expectations = []
if args.expectations != None:
    expectations = args.expectations
else: 
    qubits = [0,1,2,3]
    for q in qubits:
        expect = "I"*4
        expect = expect[:q] + 'Z' + expect[q+1:]
        expectations.append("".join(reversed(expect)))
noise_strengths = [0.5,1,2]
if args.noise_strengths != None:
	noise_strengths = args.noise_strengths
print(persamples, noise_strengths, expectations)