"""
__author__      = "Shivvrat Arya"
__version__     = "Python3.7"
"""
from numpy import array


def get_uai_data(file_name):
	"""
	Creates all the variables that will be needed for variable elimination method from the uai file
	:param file_name: THis is the filename in which the dataset is present
	:return: all the variables needed for the variable elimination
	"""
	try:
		with open(file_name) as FileObj:
			# Initialize the variables need
			preamble = False
			markov = False
			function_table = False
			count_for_preamble = 3
			table_start = False
			function_number = 0
			count_for_clique = 0
			num_of_var_in_clique = []
			var_in_clique = []
			# Looping for each line
			for lines in FileObj.readlines():
				# Using the constraints given in the file to get the values
				if 'c' in lines[0] or lines == '\n':
					continue
				if 'MARKOV' in lines.upper():
					preamble = True
					markov = True
					continue
				if "BAYES" in lines.upper():
					preamble = True
					markov = False
					continue
				if preamble and count_for_preamble == 3:
					num_of_var = list(map(int, lines.split()))[0]
					count_for_preamble -= 1
					continue
				if preamble and count_for_preamble == 2:
					cardinalities = list(map(int, lines.split()))
					count_for_preamble -= 1
					continue
				if preamble and count_for_preamble == 1:
					num_of_cliques = list(map(int, lines.split()))[0]
					distribution_array = list()
					count_for_preamble -= 1
					count_for_clique = num_of_cliques
					continue

				if preamble and count_for_preamble == 0 and count_for_clique >= 0:
					all_values = list(map(int, lines.split()))
					num_of_var_in_clique.append(all_values[0])
					var_in_clique.append(all_values[1:])
					count_for_clique -= 1
					if count_for_clique == 0:
						preamble = not preamble
						function_table = not function_table
						num_of_cliques_count = num_of_cliques
						function_number = 0
					continue
				if function_table and num_of_cliques_count > 0:
					order = 1
					table_start = True
					function_table = not function_table
					for each in var_in_clique[function_number]:
						order = order * cardinalities[each]
					splitted_line = list(map(int, lines.split()))
					if splitted_line[0] == order:
						current_table = []
					continue
				if table_start and num_of_cliques_count > 0:
					new_values = list(map(float, lines.split()))
					for each_value in new_values:
						current_table.append(each_value)
						# distribution_array[function_number].append(each_value)
						order -= 1
					if order == 0:
						table_start = not table_start
						function_table = not function_table
						function_number = function_number + 1
						distribution_array.append(current_table)
						current_table = []
		return num_of_var, array(cardinalities), num_of_cliques, array(num_of_var_in_clique), array(
				var_in_clique), array(
				distribution_array), markov
	except:
		print("The directory for the uai file is not correct, please check.")
		exit(0)


if __name__ == "__main__":
	print(get_uai_data("hw5-data\\dataset1\\1.uai"))
