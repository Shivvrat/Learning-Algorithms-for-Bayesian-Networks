"""
Get the input data given the directory
__author__      = "Shivvrat Arya"
__version__     = "Python3.7"
"""
from numpy import array


def get_input(directory):
	# The not seen data is represented with -1
	first_line = True
	list_example = []
	with open(directory) as FileObj:
		print(directory)
		for lines in FileObj.readlines():
			if first_line:
				num_var, num_examples = map(int, lines.split())
				first_line = not first_line
			elif lines.isspace():
				continue
			else:
				next_example = list(map(str, lines.split()))
				next_example = [x if (x.isdigit()) else -1 for x in next_example]
				list_example.append(list(map(int, next_example)))
	examples = array(list_example)
	return examples, num_var, num_examples
