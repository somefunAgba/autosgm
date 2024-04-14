# For relative imports to work in Python 3.6

import sys, os
from pathlib import Path

# sys.path[0] = str(Path(__file__).parents[0])
# print(sys.path[0])

assert sys.path[0] == str(Path(__file__).parents[0])

pass