import argparse
import sys
import warnings

from numpy import mean, array, var

import fod_learn
import get_data_uai
import get_input_examples
import pod_em_learn
from log_liklihood import find_log_liklihood_for_one_example, true_log_liklihood_per_example, \
	find_log_liklihood_for_one_example_mixture_model
from mixture_random_bayes import train

warnings.filterwarnings("ignore")


# For example to run the task 1 algorithm train file -  train-f-1.txt, test file - test.txt, uai file - 1.uai,
# you can run the following code
# python main.py --uai_file 1.uai --task_id 1 --training_data train-f-1.txt --test_data test.txt

def parse(argv):
	"""
    Usage -> python3 train.py --lr 1e-3 --batch_size 16 --epochs 20
    """
	parser = argparse.ArgumentParser()
	parser.add_argument('--uai_file',
	                    type = str,
	                    help = 'give uai file',
	                    default = "")

	parser.add_argument('--task_id',
	                    type = int,
	                    help = 'the algorithm to use',
	                    default = 0)

	parser.add_argument('--training_data',
	                    type = str,
	                    help = 'give training data file',
	                    default = "")

	parser.add_argument('--test_data',
	                    type = str,
	                    help = 'give test data file',
	                    default = "")

	return parser.parse_args(argv)


def main(args):
	"""
    This is the main function which is used to run all the algorithms
    :return:
    """
	print_arguments_given(args)
	# please change the value of the number of iterations for EM by using the below line
	num_of_iter_for_EM = 20
	if args.task_id == 1:
		examples, num_var, num_examples = get_input_examples.get_input(args.training_data)
		test_examples, num_var, num_examples = get_input_examples.get_input(args.test_data)
		num_of_var, cardinalities, num_of_cliques, num_of_var_in_clique, var_in_clique, distribution_array, \
		markov = get_data_uai.get_uai_data(args.uai_file)
		parameters = fod_learn.train(examples, var_in_clique, markov, cardinalities)
		count = 0
		ll_diff = 0
		for each_example in test_examples:
			count += 1
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

	elif args.task_id == 2:
		ll_diff_main = []
		for each in range(5):
			examples, num_var, num_examples = get_input_examples.get_input(args.training_data)
			test_examples, num_var, num_examples = get_input_examples.get_input(args.test_data)
			num_of_var, cardinalities, num_of_cliques, num_of_var_in_clique, var_in_clique, distribution_array, \
			markov = get_data_uai.get_uai_data(args.uai_file)
			parameters = pod_em_learn.train(examples, var_in_clique, markov, cardinalities, num_of_iter_for_EM)
			ll_diff = 0
			count = 0
			for each_example in test_examples:
				count += 1
				log_liklihood = find_log_liklihood_for_one_example(each_example, var_in_clique, parameters,
				                                                   cardinalities)
				true_log_liklihood = true_log_liklihood_per_example(each_example, var_in_clique, distribution_array,
				                                                    cardinalities)
				ll_diff += abs(log_liklihood - true_log_liklihood)
			ll_diff_main.append(ll_diff)
		mean_ll_diff = mean(ll_diff_main)
		std = var(ll_diff_main)
		print("____________________________")
		print("actual log likelihood difference = ", mean_ll_diff, "±", std)
		print("____________________________")
		ll_diff_main = array(ll_diff_main)
		ll_diff_main = ll_diff_main / (count * num_of_var)
		mean_ll_diff = mean(ll_diff_main)
		std = var(ll_diff_main)
		print("____________________________")
		print("log likelihood difference divided by number of examples and number of variables = ", mean_ll_diff, "±",
		      std)
		print("____________________________")
	elif args.task_id == 3:
		print("Please provide the value of k")
		try:
			k = int(input())
		except:
			print("Please give a integer value for k")
		ll_diff_main = []
		for each in range(5):
			examples, num_var, num_examples = get_input_examples.get_input(args.training_data)
			test_examples, num_var, num_examples = get_input_examples.get_input(args.test_data)
			num_of_var, actual_cardinalities, num_of_cliques, num_of_var_in_clique, actual_var_in_clique, \
			distribution_array, \
			markov = get_data_uai.get_uai_data(args.uai_file)
			parameters_for_network, parameters_for_k, var_in_clique = train(examples, list(range(num_var)),
			                                                                actual_cardinalities, k,
			                                                                num_of_iter_for_EM)
			ll_diff = 0
			count = 0
			for each_example in test_examples:
				count += 1
				# if count % 100000 == 0:
				# 	print("Doing for example,", count)
				# # break
				log_liklihood = find_log_liklihood_for_one_example_mixture_model(each_example, var_in_clique,
				                                                                 parameters_for_network,
				                                                                 parameters_for_k,
				                                                                 actual_cardinalities)
				true_log_liklihood = true_log_liklihood_per_example(each_example, actual_var_in_clique,
				                                                    distribution_array,
				                                                    actual_cardinalities)
				ll_diff += abs(log_liklihood - true_log_liklihood)
			ll_diff_main.append(ll_diff)
		mean_ll_diff = mean(ll_diff_main)
		std = var(ll_diff_main)
		print("____________________________")
		print("actual log likelihood difference = ", mean_ll_diff, "±", std)
		print("____________________________")
		ll_diff_main = array(ll_diff_main)
		ll_diff_main = ll_diff_main / (count * num_of_var)
		mean_ll_diff = mean(ll_diff_main)
		std = var(ll_diff_main)
		print("____________________________")
		print("log likelihood difference divided by number of examples and number of variables = ", mean_ll_diff, "±",
		      std)
		print("____________________________")
	else:
		print("We can only do 3 tasks as of now")


def print_arguments_given(args):
	"""
	This function is used to print the arguments given in command
	:param args: The arguments
	:return: N/A
	"""
	print('=' * 100)
	print('Uai file                     : {}'.format(args.uai_file))
	print('Task number                  : {}'.format(args.task_id))
	print('Training data file           : {}'.format(args.training_data))
	print('Test data file               : {}'.format(args.test_data))
	print('=' * 100)


if __name__ == '__main__':
	main(parse(sys.argv[1:]))
