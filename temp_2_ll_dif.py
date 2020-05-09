"""

__author__      = "Shivvrat Arya"
__version__     = "Python3.7"
"""

import fod_learn
import get_data_uai
import get_input_examples
from log_liklihood import find_log_liklihood_for_one_example, true_log_liklihood_per_example

# examples_with_weights, num_var, num_examples = get_input_examples.get_input("hw5-data\\my\\temp.txt")
train_examples, num_var, num_examples = get_input_examples.get_input("hw5-data\\dataset1\\train-f-1.txt")
test_examples, num_var, num_examples = get_input_examples.get_input("hw5-data\\dataset1\\test.txt")
num_of_var, cardinalities, num_of_cliques, num_of_var_in_clique, var_in_clique, distribution_array, \
markov = get_data_uai.get_uai_data("hw5-data\\dataset1\\1.uai")

# num_of_var, cardinalities, num_of_cliques, num_of_var_in_clique, var_in_clique, distribution_array, \
# markov = get_data_uai.get_uai_data("hw5-data\\my\\1.uai")
parameters = fod_learn.train(train_examples, var_in_clique, markov, cardinalities)
# log_liklihood = 0
# true_log_liklihood = 0
ll_diff = 0
count = 0
for each_example in test_examples:
	count += 1
	if count % 10000 == 0:
		print("Doing for  example", count)
	log_liklihood = find_log_liklihood_for_one_example(each_example, var_in_clique, parameters, cardinalities)
	true_log_liklihood = true_log_liklihood_per_example(each_example, var_in_clique, distribution_array,
	                                                    cardinalities)
	ll_diff += abs(log_liklihood - true_log_liklihood)

ll_diff = ll_diff / (count * num_of_var)
print("____________________________")
print("log likelihood difference = ", ll_diff)
print("____________________________")
