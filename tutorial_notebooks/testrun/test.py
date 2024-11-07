import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--circuitfilename', type=str, help='Set the name of the file, that contains the function making you circuit', default="circuits")
parser.add_argument('--circuitfunction', type=str, help='Set the name of the function, that return your circuits. The function can only take a backend as input and has to return an array of circuits', default="test")
args = parser.parse_args()

if args.circuitfilename.endswith(".py"):
    circuitfilename = args.circuitfilename[:-3]
else:
    circuitfilename = args.circuitfilename
import importlib.util
spec = importlib.util.spec_from_file_location(circuitfilename, f"{(circuitfilename)}.py", )
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
get_circuit = getattr(module, args.circuitfunction)
circuits = get_circuit("backend")
    