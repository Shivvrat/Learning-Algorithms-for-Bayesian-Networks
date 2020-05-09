"""
__author__      = "Shivvrat Arya"
__version__     = "Python3.7"
"""
import sys
from copy import deepcopy
from itertools import product as iter_product

from more_itertools import locate
from numpy import prod, product, random, put, sum, divide, array, log10


# todo check if both the type complete and incomplete examples_with_weights give same format
def get_index_given_truth_values(variables, truth_values, cardinalities):
	"""
	The function converts truth table tuples to array indices
	:param variables: The variable in factor
	:param truth_values: The values of given variables in the truth table (same order as variable)
	:param cardinalities: The cardinality of the given variables (same order as variable)
	:return:The index for the array
	"""
	index = 0

	number = 0
	while number < len(variables):
		number_of_tuples_for_current_var = prod(cardinalities[number + 1:]) * truth_values[number]
		index = index + number_of_tuples_for_current_var
		number += 1
	return int(index)


def get_truth_values_given_index(variables, index_value, cardinalities):
	"""
	Gives the truth value (values in tuple) for given index of the array
	@param variables: list containing all the variables in the graph which are in the factor
	@param index_value: Index value for which the tuple is to be founded
	@param cardinalities: Cardinalities of the given variables
	@return: the tuple for corresponding index value
	"""
	number = 0
	truth_table_value = []
	while number < len(variables):
		truth_table_value.append(int(index_value // prod(cardinalities[number + 1:])))
		index_value = index_value - (index_value // prod(cardinalities[number + 1:])) * prod(
				cardinalities[number + 1:])
		number += 1
	return truth_table_value


def generate_random_parameters(var_in_clique, cardinalities_of_var):
	parameters = []
	cardinalities_of_var = array(cardinalities_of_var)
	var_in_clique_np = array(var_in_clique)
	for each_clique in range(len(var_in_clique)):
		var_in_this_clique = var_in_clique_np[each_clique]
		cardinalities_of_var_in_this_clique = cardinalities_of_var[var_in_this_clique]
		total_tuples = product(cardinalities_of_var_in_this_clique)
		if len(var_in_this_clique) > 1:
			# indexing of parameters is done on the basis of parent variables only
			number_of_parameters = int(total_tuples / cardinalities_of_var_in_this_clique[-1])
		else:
			number_of_parameters = 1
		parameters_for_this_clique = random.rand(number_of_parameters)
		if len(parameters_for_this_clique) == 1:
			parameters.append(float(parameters_for_this_clique))
		else:
			parameters.append(parameters_for_this_clique.tolist())
	return parameters


def complete_data(examples, cardinalities_of_var, parameters, var_in_clique):
	final_examples = []
	final_weights = []
	for each_example in examples:
		if -1 not in each_example:
			weight = 1
			final_examples.append(each_example)
			final_weights.append(weight)
		elif -1 in each_example:
			weights_for_completed_examples = []
			completed_examples = []

			var_not_present = list(locate(each_example, lambda a: a == -1))
			cardinalities_of_var = array(cardinalities_of_var)
			cardinalities_of_var_not_present = cardinalities_of_var[var_not_present]
			all_possible_completions = []
			for each_var in range(len(var_not_present)):
				completions_for_var_not_present = list(range(cardinalities_of_var_not_present[each_var]))
				all_possible_completions.append(completions_for_var_not_present)
			all_completions_only_missing_var = list(iter_product(*all_possible_completions))
			for each_tuple_missing_var in all_completions_only_missing_var:
				temp_example = deepcopy(each_example)
				put(temp_example, var_not_present, each_tuple_missing_var)
				weight_for_current_example = 1
				for each_clique in range(len(var_in_clique)):
					if compare_list(var_not_present, var_in_clique[each_clique]):
						values_for_current_clique = temp_example[var_in_clique[each_clique]]
						if len(var_in_clique[each_clique]) > 1:
							index_for_current_example = get_index_given_truth_values(var_in_clique[each_clique][:-1],
							                                                         values_for_current_clique[:-1],
							                                                         cardinalities_of_var[
								                                                         var_in_clique[each_clique][
								                                                         :-1]])
							if values_for_current_clique[-1] == 0:
								weight_for_current_example *= parameters[each_clique][index_for_current_example]
							elif values_for_current_clique[-1] == 1:
								weight_for_current_example *= (1 - parameters[each_clique][index_for_current_example])
							else:
								print("Error")
						elif len(var_in_clique[each_clique]) == 1:
							if values_for_current_clique[0] == 0:
								weight_for_current_example *= parameters[each_clique]
							elif values_for_current_clique[0] == 1:
								weight_for_current_example *= (1 - parameters[each_clique])
							else:
								print("Error")
				weights_for_completed_examples.append(weight_for_current_example)
				completed_examples.append(temp_example)
			denom = sum(weights_for_completed_examples)
			weights_for_completed_examples = divide(weights_for_completed_examples, denom)
			for weight, example in zip(weights_for_completed_examples, completed_examples):
				final_examples.append(example)
				final_weights.append(float(weight))
	return final_examples, final_weights


def compare_list(list_1, list_2):
	for each in list_2:
		if each in list_1:
			return True
	return False


def log_sum_exp(a):
	"""
	Uses the log sum exp technique to do computation in higher domain than float
	@param a: The array for which we want to find the sum
	@return: The sum of the array founded by log sum exp technique
	"""
	# return logsum1(a)/log(10)
	mx = max(a)
	to_return = log10(sum([threshold(10 ** (i - mx)) for i in a])) + mx
	return to_return


def threshold(x):
	"""
	This function is used to find the value in the given limit
	@param x: The value which needs to be checked
	@return: the value according to the threshold
	"""
	if x == float('inf'):
		return sys.float_info.max
	return x
