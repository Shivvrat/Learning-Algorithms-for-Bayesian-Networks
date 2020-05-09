"""

__author__      = "Shivvrat Arya"
__version__     = "Python3.7"
"""
from copy import deepcopy
from random import choice, randint, sample

from numpy import array, setdiff1d, concatenate, reshape, append, put, divide, nan_to_num, product, zeros
from numpy.random import rand

from helper import generate_random_parameters, get_index_given_truth_values


def create_DAG(variables):
	list_of_var = deepcopy(variables)
	done_var = []
	var_in_clique = []
	while len(list_of_var) > 0:
		child = [choice(list_of_var)]
		list_of_var = setdiff1d(list_of_var, child)
		if len(list_of_var) == len(variables) - 1:
			number_of_parents = randint(1, min(3, len(list_of_var)))
			parents = list(set(sample(list(list_of_var), number_of_parents)))
			var_in_clique += (list([each] for each in parents))
		else:
			number_of_parents = randint(1, min(3, len(done_var)))
			parents = list(set(sample(done_var, number_of_parents)))
		list_of_var = setdiff1d(list_of_var, parents)
		done_var += parents
		done_var += child
		var_in_this_clique = parents + child
		var_in_clique.append(array(var_in_this_clique))
	return var_in_clique


def generate_parameters(cardinalities_of_var, value_of_k, var_in_clique):
	parameters_for_k = rand(value_of_k)
	parameters_for_k = parameters_for_k / sum(parameters_for_k)
	parameters_for_network = []
	for each_value_of_k in range(value_of_k):
		parameters = generate_random_parameters(var_in_clique[each_value_of_k], cardinalities_of_var)
		parameters_for_network.append(parameters)
	return parameters_for_network, parameters_for_k


def add_latent_var(examples, value_of_k, cardinalities_of_var):
	#  adding latent var at the end
	row_for_latent_var = [-1] * len(examples)
	column_for_latent_var = reshape(row_for_latent_var, (-1, 1))
	pod_examples = concatenate((examples, column_for_latent_var), 1)
	# the value of k is added at last
	cardinalities_of_var = append(cardinalities_of_var, value_of_k)
	return pod_examples, cardinalities_of_var


def train(examples, variables, cardinalities_of_var, value_of_k, iterations):
	var_in_clique = []
	for each_k in range(value_of_k):
		var_in_clique.append(create_DAG(variables))
	parameters_for_network, parameters_for_k = generate_parameters(cardinalities_of_var, value_of_k,
	                                                               var_in_clique)
	pod_examples, cardinalities_of_var = add_latent_var(examples, value_of_k, cardinalities_of_var)
	for each in range(iterations):
		weighted_examples, weights = e_step(parameters_for_network, parameters_for_k, pod_examples, var_in_clique,
		                                    cardinalities_of_var)
		parameters_for_network, parameters_for_k = m_step(weighted_examples, weights, var_in_clique,
		                                                  cardinalities_of_var, value_of_k)
	return parameters_for_network, parameters_for_k, var_in_clique


def e_step(parameters_for_network, parameters_for_k, examples, var_in_clique, cardinalities_of_var):
	weighted_examples, weights = complete_data(examples, cardinalities_of_var, parameters_for_network,
	                                           parameters_for_k,
	                                           var_in_clique)
	return weighted_examples, weights


def m_step(examples, weights, var_in_clique, cardinalities_of_var, value_of_k):
	parameters_for_network, parameters_for_k = get_parameters_given_weighted_example(examples, weights, var_in_clique,
	                                                                                 cardinalities_of_var, value_of_k)
	return parameters_for_network, parameters_for_k


def complete_data(examples, cardinalities_of_var, parameters_for_network, parameters_for_k, var_in_clique):
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
			cardinalities_of_var = array(cardinalities_of_var)
			var_not_present = len(cardinalities_of_var) - 1
			cardinalities_of_var_not_present = cardinalities_of_var[var_not_present]
			completions_for_var_not_present = list(range(cardinalities_of_var_not_present))
			for each_tuple_missing_var in completions_for_var_not_present:
				temp_example = deepcopy(each_example)
				put(temp_example, var_not_present, each_tuple_missing_var)
				weight_for_current_example = 1
				value_of_k = temp_example[-1]
				# for each tuple we have diff network thus weight will be different
				for each_clique in range(len(var_in_clique[each_tuple_missing_var])):
					if True:
						values_for_current_clique = temp_example[var_in_clique[each_tuple_missing_var][each_clique]]
						if len(var_in_clique[each_tuple_missing_var][each_clique]) > 1:
							index_for_current_example = get_index_given_truth_values(
									var_in_clique[each_tuple_missing_var][each_clique][:-1],
									values_for_current_clique[:-1],
									cardinalities_of_var[
										var_in_clique[each_tuple_missing_var][each_clique][
										:-1]])
							if values_for_current_clique[-1] == 0:
								weight_for_current_example *= parameters_for_network[value_of_k][each_clique][
									index_for_current_example]
							elif values_for_current_clique[-1] == 1:
								weight_for_current_example *= (1 - parameters_for_network[value_of_k][each_clique][
									index_for_current_example])
							else:
								print("Error")
						elif len(var_in_clique[each_tuple_missing_var][each_clique]) == 1:
							if values_for_current_clique[0] == 0:
								weight_for_current_example *= parameters_for_network[value_of_k][each_clique]
							elif values_for_current_clique[0] == 1:
								weight_for_current_example *= (1 - parameters_for_network[value_of_k][each_clique])
							else:
								print("Error")
				weight_for_current_example *= parameters_for_k[value_of_k]
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


def get_parameters_given_weighted_example(examples, weights, var_in_clique, cardinalities_of_var, number_of_k):
	parameters_for_network = [[] for i in range(number_of_k)]
	parameters_for_k = []
	# todo change the use of var_in_clique for each val of k
	cardinalities_of_var = array(cardinalities_of_var)
	examples = array(examples)
	var_in_clique_np = array(var_in_clique)
	for each_val_of_k in range(number_of_k):
		for each_clique in range(len(var_in_clique[each_val_of_k])):
			var_in_this_clique = var_in_clique_np[each_val_of_k][each_clique]
			vals_for_this_clique = examples[:, var_in_this_clique]
			val_of_k = examples[:, -1]
			cardinalities_of_var_in_this_clique = cardinalities_of_var[var_in_this_clique]
			total_tuples = product(cardinalities_of_var_in_this_clique)
			if len(var_in_this_clique) > 1:
				number_of_parameters = int(total_tuples / cardinalities_of_var_in_this_clique[-1])
				nums = zeros([number_of_parameters, ])
				denom = zeros([number_of_parameters, ])
				for each_vals_for_this_clique, weights_for_this_example, value_of_k_for_this_example in zip(
						vals_for_this_clique, weights, val_of_k):
					if value_of_k_for_this_example == each_val_of_k:
						# indexing of parameters_for_network is done on the basis of parent variables only
						index_val_without_child = get_index_given_truth_values(var_in_this_clique[:-1],
						                                                       each_vals_for_this_clique[:-1],
						                                                       cardinalities_of_var_in_this_clique[
						                                                       :-1])
						denom[index_val_without_child] += weights_for_this_example
						if each_vals_for_this_clique[-1] == 0:
							nums[index_val_without_child] += weights_for_this_example
				parameter_val = divide(nums, denom)
				parameter_val = nan_to_num(parameter_val)
				parameter_val[parameter_val == 0] = pow(10, -5)
				parameter_val[parameter_val == 1] = 1 - pow(10, -5)
				parameters_for_network[each_val_of_k].append(parameter_val)
			else:
				count_of_zero = 0
				total_count = 0
				for each_vals_for_this_clique, weights_for_this_example, value_of_k_for_this_example in zip(
						vals_for_this_clique, weights, val_of_k):
					if value_of_k_for_this_example == each_val_of_k:
						total_count += weights_for_this_example
						if each_vals_for_this_clique[0] == 0:
							count_of_zero += weights_for_this_example
				parameter = nan_to_num(divide(count_of_zero, total_count))
				if parameter == 0:
					parameter = pow(10, -5)
				elif parameter == 1:
					parameter = 1 - pow(10, -5)
				parameters_for_network[each_val_of_k].append(parameter)
	# parameters_for_network = array(parameters_for_network)
	nums_for_k = zeros([number_of_k, ])
	denom_for_k = 0
	for each_val_of_k, weight_for_k in zip(examples[:, -1], weights):
		nums_for_k[each_val_of_k] += weight_for_k
		denom_for_k += weight_for_k
	parameters_for_k = divide(nums_for_k, denom_for_k)
	return parameters_for_network, parameters_for_k
