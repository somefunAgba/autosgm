# For relative imports to work in Python 3.6
# import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import os, sys

from pathlib import Path
sys.path[0] = str(Path(__file__).parents[0])

# if os.path.exists('__pycache__'): os.remove('__pycache__')
