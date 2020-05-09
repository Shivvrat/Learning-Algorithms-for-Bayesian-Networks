"""

__author__      = "Shivvrat Arya"
__version__     = "Python3.7"
"""

from numpy import array, zeros, nan_to_num, divide, product

import helper
from helper import generate_random_parameters, complete_data


def train(pod_examples, var_in_clique, markov, cardinalities_of_var, iterations):
	"""

	:param weighted_examples:
	:param var_in_clique:
	:param markov:
	:param cardinalities_of_var:
	:return: parameters in double form
	"""
	if not markov:
		parameters = generate_random_parameters(var_in_clique, cardinalities_of_var)
		for each in range(iterations):
			weighted_examples, weights = e_step(parameters, pod_examples, var_in_clique, cardinalities_of_var)
			parameters = m_step(weighted_examples, weights, var_in_clique, cardinalities_of_var)
		return parameters
	else:
		print("Please provide a Bayesian Network")
		print("We are doing Bayesian Learning")
		exit()


def e_step(parameters, examples, var_in_clique, cardinalities_of_var):
	weighted_examples, weights = complete_data(examples, cardinalities_of_var, parameters, var_in_clique)
	return weighted_examples, weights


def m_step(examples, weights, var_in_clique, cardinalities_of_var):
	parameters = get_parameters_given_weighted_example(examples, weights, var_in_clique, cardinalities_of_var)
	return parameters


def get_parameters_given_weighted_example(examples, weights, var_in_clique, cardinalities_of_var):
	parameters = []
	cardinalities_of_var = array(cardinalities_of_var)
	examples = array(examples)
	var_in_clique_np = array(var_in_clique)
	for each_clique in range(len(var_in_clique)):
		var_in_this_clique = var_in_clique_np[each_clique]
		vals_for_this_clique = examples[:, var_in_this_clique]
		cardinalities_of_var_in_this_clique = cardinalities_of_var[var_in_this_clique]
		total_tuples = product(cardinalities_of_var_in_this_clique)
		if len(var_in_this_clique) > 1:
			number_of_parameters = int(total_tuples / cardinalities_of_var_in_this_clique[-1])
			nums = zeros([number_of_parameters, ])
			denom = zeros([number_of_parameters, ])
			for each_vals_for_this_clique, weights_for_this_example in zip(vals_for_this_clique, weights):
				# indexing of parameters is done on the basis of parent variables only
				index_val_without_child = helper.get_index_given_truth_values(var_in_this_clique[:-1],
				                                                              each_vals_for_this_clique[:-1],
				                                                              cardinalities_of_var_in_this_clique[
				                                                              :-1])
				denom[index_val_without_child] += weights_for_this_example
				if each_vals_for_this_clique[-1] == 0:
					nums[index_val_without_child] += weights_for_this_example
			parameter_val = divide(nums, denom)
			parameter_val = nan_to_num(parameter_val)
			parameter_val[parameter_val == 0] = pow(10, -5)
			parameters.append(parameter_val)
		else:
			count_of_zero = 0
			total_count = 0
			for each_vals_for_this_clique, weights_for_this_example in zip(vals_for_this_clique, weights):
				total_count += weights_for_this_example
				if each_vals_for_this_clique[0] == 0:
					count_of_zero += weights_for_this_example
			parameter = nan_to_num(divide(count_of_zero, total_count))
			if parameter == 0:
				parameter += pow(10, -5)
			parameters.append(parameter)
	return parameters
