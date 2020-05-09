"""
__author__      = "Shivvrat Arya"
__version__     = "Python3.7"
"""
from numpy import product, array, unique, zeros, divide, nan_to_num

# storing values for parameters when child = 0
import helper


def train(examples, var_in_clique, markov, cardinalities_of_var):
	"""

	:param examples:
	:param var_in_clique:
	:param markov:
	:param cardinalities_of_var:
	:return: parameters in double form
	"""
	parameters = []
	cardinalities_of_var = array(cardinalities_of_var)
	if not markov:
		var_in_clique_np = array(var_in_clique)
		for each_clique in range(len(var_in_clique)):
			var_in_this_clique = var_in_clique_np[each_clique]
			vals_for_this_clique = examples[:, var_in_this_clique]
			cardinalities_of_var_in_this_clique = cardinalities_of_var[var_in_this_clique]
			total_tuples = product(cardinalities_of_var_in_this_clique)
			unique_vals, counts = unique(vals_for_this_clique, return_counts = True, axis = 0)
			if len(var_in_this_clique) > 1:
				# indexing of parameters is done on the basis of parent variables only
				number_of_parameters = int(total_tuples / cardinalities_of_var_in_this_clique[-1])
				# Using 1-Laplace Smoothing
				nums = zeros([number_of_parameters, ]) + 1
				denom = zeros([number_of_parameters, ]) + cardinalities_of_var_in_this_clique[-1]
				# todo need to use parent only for parameters
				for each_tuple in zip(unique_vals, counts):
					index_val_without_child = helper.get_index_given_truth_values(var_in_this_clique[:-1],
					                                                              each_tuple[0][:-1],
					                                                              cardinalities_of_var_in_this_clique[
					                                                              :-1])
					denom[index_val_without_child] += each_tuple[1]
					temp = each_tuple[0][-1]
					if each_tuple[0][-1] == 0:
						nums[index_val_without_child] += each_tuple[1]
				parameter_val = divide(nums, denom)
				parameter_val = nan_to_num(parameter_val)
				parameters.append(list(parameter_val))
			else:
				count_of_zero = 0
				total_count = 0
				for each_tuple in zip(unique_vals, counts):
					each_tuple = array(each_tuple)
					vals_for_this_clique = int(each_tuple[0])
					total_count += each_tuple[1]
					if vals_for_this_clique == 0:
						count_of_zero += each_tuple[1]
				parameters.append(nan_to_num(divide(count_of_zero, total_count)))
	return parameters
