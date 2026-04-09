import json
import re
from pathlib import Path
from typing import Dict, List



from opencompass.datasets.math import MATHEvaluator

# ---------Below is the test section for the class------------
evaluator = MATHEvaluator(version='v2')

print(evaluator)

predicted_answer = "[\n\\boxed{25}\n\\]"

# Standard answer
standard_answer = "25"

# Use is_equiv function to determine answer equivalence
result = evaluator.is_equiv(predicted_answer, standard_answer)

print(result)  # True means equivalent, False means not equivalent
