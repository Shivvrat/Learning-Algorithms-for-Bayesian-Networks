"""

__author__      = "Shivvrat Arya"
__version__     = "Python3.7"
"""
# 4,6,4,6

import get_data_uai
import get_input_examples
from log_liklihood import true_log_liklihood_per_example, find_log_liklihood_for_one_example_mixture_model
from mixture_random_bayes import train

to_train = ["hw5-data\\dataset3\\train-f-1.txt", "hw5-data\\dataset3\\train-f-2.txt",
            "hw5-data\\dataset3\\train-f-3.txt",
            "hw5-data\\dataset3\\train-f-4.txt"]

for each in to_train:
	# examples, num_var, num_examples = get_input_examples.get_input("hw5-data\\my\\temp.txt")
	examples, num_var, num_examples = get_input_examples.get_input(each)
	test_examples, _, _ = get_input_examples.get_input("hw5-data\\dataset3\\test.txt")
	# num_of_var, actual_cardinalities, num_of_cliques, num_of_var_in_clique, actual_var_in_clique,
	# distribution_array, \
	# markov = get_data_uai.get_uai_data("hw5-data\\my\\1.uai")
	num_of_var, actual_cardinalities, num_of_cliques, num_of_var_in_clique, actual_var_in_clique, distribution_array, \
	markov = get_data_uai.get_uai_data("hw5-data\\dataset3\\3.uai")
	parameters_for_network, parameters_for_k, var_in_clique = train(examples, list(range(num_var)),
	                                                                actual_cardinalities, 6, 20)
	ll_diff = 0
	count = 0
	for each_example in test_examples:
		count += 1
		# if count % 100000 == 0:
		# 	print("Doing for example,", count)
		# # break
		log_liklihood = find_log_liklihood_for_one_example_mixture_model(each_example, var_in_clique,
		                                                                 parameters_for_network, parameters_for_k,
		                                                                 actual_cardinalities)
		true_log_liklihood = true_log_liklihood_per_example(each_example, actual_var_in_clique, distribution_array,
		                                                    actual_cardinalities)
		ll_diff += abs(log_liklihood - true_log_liklihood)
	print("____________________________")
	print("actual log likelihood difference = ", ll_diff)
	print("____________________________")
	ll_diff = ll_diff / (count * num_of_var)
	print("____________________________")
	print("log likelihood difference divided by number of examples and number of variables = ", ll_diff)
	print("____________________________")
