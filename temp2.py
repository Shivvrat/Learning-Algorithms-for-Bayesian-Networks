"""

__author__      = "Shivvrat Arya"
__version__     = "Python3.7"
"""
import get_data_uai
import get_input_examples_partially_observed
import pod_em_learn
from log_liklihood import find_log_liklihood_for_one_example, true_log_liklihood_per_example

examples, _, _ = get_input_examples_partially_observed.get_input("hw5-data\\dataset1\\train-p-2.txt")
# examples, num_var, num_examples = get_input_examples_partially_observed.get_input("hw5-data\\my\\pod.txt")
test_examples, _, _ = get_input_examples_partially_observed.get_input("hw5-data\\dataset1\\test.txt")
# test_examples, num_var, num_examples = get_input_examples_partially_observed.get_input("hw5-data\\my\\pod1.txt")
num_of_var, cardinalities, num_of_cliques, num_of_var_in_clique, var_in_clique, distribution_array, \
markov = get_data_uai.get_uai_data("hw5-data\\dataset1\\1.uai")
# num_of_var, cardinalities, num_of_cliques, num_of_var_in_clique, var_in_clique, distribution_array, \
# markov = get_data_uai.get_uai_data("hw5-data\\my\\1.uai")
parameters = pod_em_learn.train(examples, var_in_clique, markov, cardinalities, 20)
ll_diff = 0
count = 0
for each_example in test_examples:
	count += 1
	if count % 20000 == 0:
		print("Doing for example,", count)
	log_liklihood = find_log_liklihood_for_one_example(each_example, var_in_clique, parameters, cardinalities)
	true_log_liklihood = true_log_liklihood_per_example(each_example, var_in_clique, distribution_array,
	                                                    cardinalities)
	ll_diff += abs(log_liklihood - true_log_liklihood)

print("____________________________")
print("actual log likelihood difference = ", ll_diff)
print("____________________________")
ll_diff = ll_diff / (count * num_of_var)
print("____________________________")
print("log likelihood difference divided by number of examples and number of variables = ", ll_diff)
print("____________________________")
