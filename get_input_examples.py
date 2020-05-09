"""
Get the input data given the directory
__author__      = "Shivvrat Arya"
__version__     = "Python3.7"
"""
from numpy import array


def get_input(directory):
	first_line = True
	list_example = []
	with open(directory) as FileObj:
		for lines in FileObj.readlines():
			if first_line:
				num_var, num_examples = map(int, lines.split())
				first_line = not first_line
			elif lines.isspace():
				continue
			else:
				list_example.append(array(list(map(int, lines.split()))))
	examples = array(list_example)
	return examples, num_var, num_examples
