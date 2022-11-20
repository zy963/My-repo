############ Welcome to HW7 ############
# TODO: Andrew-id: zheyil


# Imports
# Don't import any other library
import numpy as np
from utils import make_dict, parse_file, get_matrices, write_predictions, write_metrics
import argparse
import logging

# Setting up the argument parser
# don't change anything here
parser = argparse.ArgumentParser()
parser.add_argument('validation_input', type=str,
                    help='path to validation input .txt file')
parser.add_argument('index_to_word', type=str,
                    help='path to index_to_word.txt file')
parser.add_argument('index_to_tag', type=str,
                    help='path to index_to_tag.txt file')
parser.add_argument('init', type=str,
                    help='path to the learned hmm_init.txt (pi) file')
parser.add_argument('emission', type=str,
                    help='path to the learned hmm_emission.txt (A) file')
parser.add_argument('transition', type=str,
                    help='path to the learned hmm_transition.txt (B) file')
parser.add_argument('prediction_file', type=str,
                    help='path to store predictions')
parser.add_argument('metric_file', type=str,
                    help='path to store metrics')                    
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


# Hint: You might find it helpful to define functions 
# that do the following:
# 1. Calculate Alphas
# 2. Calculate Betas
# 3. Implement the LogSumExpTrick
# 4. Calculate probabilities and predictions
def logsumexp(array):
    return np.max(array) + np.log(np.sum(np.exp(array - np.max(array))))


def forward(sentence, tag_num, word_dict, init, emission, transition):
    alpha = np.zeros((len(sentence), tag_num))
    for i in range(1, len(sentence) + 1):
        if i == 1:
            alpha[0] = np.log(init) + np.log(emission[:, word_dict[sentence[0]]])
        else:
            for j in range(1, tag_num + 1):
                new = np.zeros(tag_num)
                for k in range(1, tag_num + 1):
                    new[k - 1] = alpha[i - 2, k - 1] + np.log(transition[k - 1, j - 1])
                alpha[i - 1, j - 1] = np.log(emission[j - 1, word_dict[sentence[i - 1]]]) + logsumexp(new)
    return alpha


def backward(sentence, tag_num, word_dict, emission, transition):
    beta = np.zeros((len(sentence), tag_num))
    for i in range(len(sentence)):
        if i != 0:
            for j in range(tag_num):
                new = np.zeros(tag_num)
                for k in range(tag_num):
                    o = np.log(transition[j, k])
                    p = emission[k, word_dict[sentence[len(sentence) - i]]]
                    new[k] = beta[len(sentence) - i, k] + o + np.log(p)
                beta[len(sentence) - i - 1, j] = logsumexp(new)
    return beta


def forwardback(word_seq, tag_dict, tag_num, word_dict, init, emission, transition):
    alpha = forward(word_seq, tag_num, word_dict, init, emission, transition)
    beta = backward(word_seq, tag_num, word_dict, emission, transition)
    comb = alpha + beta

    tag_dict_new = {v: k for k, v in tag_dict.items()}
    prediction_tag = []
    prediction = [0] * len(word_seq)
    for i in range(len(np.argmax(comb, axis=1))):
        prediction[i] = tag_dict_new[np.argmax(comb, axis=1)[i]]
        prediction_tag.append(prediction[i])
    likelihood = logsumexp(alpha[-1])

    return prediction_tag, likelihood


# TODO: Complete the main function
def main(args):

    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    tag_dict = make_dict(args.index_to_tag)
    tag_num = len(tag_dict)
    word_num = len(word_dict)

    # Parse the validation file
    sentences, tags = parse_file(args.validation_input)

    ## Load your learned matrices
    ## Make sure you have them in the right orientation
    ## TODO:  Uncomment the following line when you're ready!
    
    init, emission, transition = get_matrices(args)

    # TODO: Conduct your inferences
    sum_llh = 0
    predicted_tags = []
    for i in range(len(sentences)):
        prediction_tags, llh = forwardback(sentences[i], tag_dict, tag_num, word_dict, init, emission, transition)
        predicted_tags.append(prediction_tags)
        sum_llh += llh

    
    # TODO: Generate your probabilities and predictions

    # predicted_tags = #TODO: store your predicted tags here (in the right order)
    avg_log_likelihood = sum_llh / len(sentences)  # TODO: store your calculated average log-likelihood here
    
    accuracy = 0 # We'll calculate this for you

    ## Writing results to the corresponding files.  
    ## We're doing this for you :)
    ## TODO: Just Uncomment the following lines when you're ready!

    accuracy = write_predictions(args.prediction_file, sentences, predicted_tags, tags)
    write_metrics(args.metric_file, avg_log_likelihood, accuracy)

    return

if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    main(args)
