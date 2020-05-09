"""

__author__      = "Shivvrat Arya"
__version__     = "Python3.7"
"""
from numpy import array
from numpy import log10

import helper


def find_log_liklihood_for_one_example_mixture_model(example, var_in_clique, parameters, parameters_for_k,
                                                     cardinalities):
	log_liklihood = []
	cardinalities = array(cardinalities)
	var_in_clique = array(var_in_clique)
	for each_val_of_k in range(len(parameters_for_k)):
		log_liklihood_for_this_val_of_k = 1
		for each_clique in range(len(var_in_clique[each_val_of_k])):
			var_in_this_clique = var_in_clique[each_val_of_k][each_clique]
			cardinalities_of_var_in_this_clique = cardinalities[var_in_this_clique]
			values_of_tuple = example[var_in_this_clique]
			if len(var_in_this_clique) > 1:
				parents = values_of_tuple[:-1]
				parents_var_in_this_clique = var_in_this_clique[:-1]
				cardinalities_of_parents_in_this_clique = cardinalities_of_var_in_this_clique[:-1]
				child = values_of_tuple[-1]
				index_for_parents = helper.get_index_given_truth_values(parents_var_in_this_clique, parents,
				                                                        cardinalities_of_parents_in_this_clique)
				if parameters[each_val_of_k][each_clique][index_for_parents] == 0:
					parameters[each_val_of_k][each_clique][index_for_parents] += pow(10, -5)
				elif parameters[each_val_of_k][each_clique][index_for_parents] == 1:
					parameters[each_val_of_k][each_clique][index_for_parents] -= pow(10, -5)
				if child == 0:
					log_liklihood_for_this_val_of_k += log10(parameters[each_val_of_k][each_clique][index_for_parents])
				elif child == 1:
					log_liklihood_for_this_val_of_k += log10(
							1 - parameters[each_val_of_k][each_clique][index_for_parents])
				else:
					print("Error line 37")
			elif len(var_in_this_clique) == 1:
				if parameters[each_val_of_k][each_clique] == 0:
					parameters[each_val_of_k][each_clique] += pow(10, -5)
				elif parameters[each_val_of_k][each_clique] == 1:
					parameters[each_val_of_k][each_clique] -= pow(10, -5)
				if values_of_tuple[0] == 0:
					log_liklihood_for_this_val_of_k += log10(parameters[each_val_of_k][each_clique])
				elif values_of_tuple[0] == 1:
					log_liklihood_for_this_val_of_k += log10(1 - parameters[each_val_of_k][each_clique])
				else:
					print("Error line 48")
			else:
				print("Error line ")
		log_liklihood_for_this_val_of_k += log10(parameters_for_k[each_val_of_k])
		log_liklihood.append(log_liklihood_for_this_val_of_k)
	log_liklihood_output = helper.log_sum_exp(log_liklihood)
	return log_liklihood_output


def find_log_liklihood_for_one_example(example, var_in_clique, parameters, cardinalities):
	log_liklihood = 0
	cardinalities = array(cardinalities)
	var_in_clique = array(var_in_clique)
	for each_clique in range(len(var_in_clique)):
		var_in_this_clique = list(var_in_clique[each_clique])
		cardinalities_of_var_in_this_clique = cardinalities[var_in_this_clique]
		values_of_tuple = example[var_in_this_clique]
		if len(var_in_this_clique) > 1:
			parents = values_of_tuple[:-1]
			parents_var_in_this_clique = var_in_this_clique[:-1]
			cardinalities_of_parents_in_this_clique = cardinalities_of_var_in_this_clique[:-1]
			child = values_of_tuple[-1]
			index_for_parents = helper.get_index_given_truth_values(parents_var_in_this_clique, parents,
			                                                        cardinalities_of_parents_in_this_clique)
			if parameters[each_clique][index_for_parents] == 0:
				parameters[each_clique][index_for_parents] += pow(10, -5)
			elif parameters[each_clique][index_for_parents] == 1:
				parameters[each_clique][index_for_parents] -= pow(10, -5)
			if child == 0:
				log_liklihood += log10(parameters[each_clique][index_for_parents])
			elif child == 1:
				log_liklihood += log10(1 - parameters[each_clique][index_for_parents])
			else:
				print("Error line 37")
		elif len(var_in_this_clique) == 1:
			if parameters[each_clique] == 0:
				parameters[each_clique] += pow(10, -5)
			elif parameters[each_clique] == 1:
				parameters[each_clique] -= pow(10, -5)
			if values_of_tuple[0] == 0:
				log_liklihood += log10(parameters[each_clique])
			elif values_of_tuple[0] == 1:
				log_liklihood += log10(1 - parameters[each_clique])
			else:
				print("Error line 48")
		else:
			print("Error line ")
	return log_liklihood


def true_log_liklihood_per_example(example, var_in_clique, true_parameters, cardinalities):
	log_liklihood = 0
	for each_clique in range(len(var_in_clique)):
		var_in_this_clique = list(var_in_clique[each_clique])
		cardinalities_of_var_in_this_clique = cardinalities[var_in_this_clique]
		values_of_tuple = example[var_in_this_clique]
		if len(var_in_this_clique) > 1:
			index = helper.get_index_given_truth_values(var_in_this_clique, values_of_tuple,
			                                            cardinalities_of_var_in_this_clique)
			values_of_conjugate_tuple = values_of_tuple
			if true_parameters[each_clique][index] == 0:
				true_parameters[each_clique][index] += pow(10, -5)
				if values_of_conjugate_tuple[-1] == 0:
					values_of_conjugate_tuple[-1] = 1
				elif values_of_conjugate_tuple[-1] == 1:
					values_of_conjugate_tuple[-1] = 0
				index_for_other_child = helper.get_index_given_truth_values(var_in_this_clique,
				                                                            values_of_conjugate_tuple,
				                                                            cardinalities_of_var_in_this_clique)
				true_parameters[each_clique][index_for_other_child] -= pow(10, -5)
			elif true_parameters[each_clique][index] == 1:
				true_parameters[each_clique][index] -= pow(10, -5)
				if values_of_conjugate_tuple[-1] == 0:
					values_of_conjugate_tuple[-1] = 1
				elif values_of_conjugate_tuple[-1] == 1:
					values_of_conjugate_tuple[-1] = 0
				index_for_other_child = helper.get_index_given_truth_values(var_in_this_clique,
				                                                            values_of_conjugate_tuple,
				                                                            cardinalities_of_var_in_this_clique)
				true_parameters[each_clique][index_for_other_child] += pow(10, -5)

			log_liklihood += log10(true_parameters[each_clique][index])
		else:
			if true_parameters[each_clique][0] == 0:
				true_parameters[each_clique][0] += pow(10, -5)
				true_parameters[each_clique][1] -= pow(10, -5)
			elif true_parameters[each_clique][0] == 1:
				true_parameters[each_clique][0] -= pow(10, -5)
				true_parameters[each_clique][1] += pow(10, -5)
			log_liklihood += log10(true_parameters[each_clique][values_of_tuple[0]])
	return log_liklihood
