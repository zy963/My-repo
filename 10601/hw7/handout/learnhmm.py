############ Welcome to HW7 ############
# TODO: Andrew-id: zheyil


# Imports
# Don't import any other library
import argparse
import numpy as np
from utils import make_dict, parse_file
import logging

# Setting up the argument parser
# don't change anything here

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .txt file')
parser.add_argument('index_to_word', type=str,
                    help='path to index_to_word.txt file')
parser.add_argument('index_to_tag', type=str,
                    help='path to index_to_tag.txt file')
parser.add_argument('init', type=str,
                    help='path to store the hmm_init.txt (pi) file')
parser.add_argument('emission', type=str,
                    help='path to store the hmm_emission.txt (A) file')
parser.add_argument('transition', type=str,
                    help='path to store the hmm_transition.txt (B) file')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


# Hint: You might find it useful to define functions that do the following:
# 1. Calculate the init matrix
# 2. Calculate the emission matrix
# 3. Calculate the transition matrix
# 4. Normalize the matrices appropriately

# TODO: Complete the main function
def main(args):

    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    tag_dict = make_dict(args.index_to_tag)

    # Parse the train file
    # Suggestion: Take a minute to look at the training file,
    # it always helps to know your data :)
    sentences, tags = parse_file(args.train_input)
    # empirical question
    sentences = sentences[0:10000]
    tags = tags[0:10000]


    logging.debug(f"Num Sentences: {len(sentences)}")
    logging.debug(f"Num Tags: {len(tags)}")
    
    
    # Train your HMM
    tag_num = len(tag_dict)
    word_num = len(word_dict)

    init = np.zeros(tag_num)
    transition = np.zeros((tag_num, tag_num))
    emission = np.zeros((tag_num, word_num))

    # init = # TODO: Construct your init matrix
    tag_init = tag_dict.copy()

    for tag in tag_init.keys():
        tag_init[tag] = 0
        for i in range(len(tags)):
            if tags[i][0] == tag:
                tag_init[tag] += 1
        tag_init[tag] += 1

    sum_init = sum(tag_init.values())
    i = 0
    for tag in tag_init.keys():
        init[i] = tag_init[tag] / sum_init
        i += 1

    # emission = # TODO: Construct your emission matrix
    for i in range(len(tags)):
        for j in range(1, len(tags[i])):
            l2_idx = tag_dict[tags[i][j]]
            l1_idx = tag_dict[tags[i][j - 1]]
            transition[l1_idx][l2_idx] += 1

    transition += 1
    for i in range(len(transition)):
        transition[i] = transition[i] / np.sum(transition[i])

    # transition = # TODO: Construct your transition matrix
    for i in range(len(tags)):
        for j in range(len(tags[i])):
            x_idx = word_dict[sentences[i][j]]
            y_idx = tag_dict[tags[i][j]]
            emission[y_idx][x_idx] += 1

    emission += 1
    for i in range(len(emission)):
        emission[i] = emission[i] / np.sum(emission[i])


    # Making sure we have the right shapes
    logging.debug(f"init matrix shape: {init.shape}")
    logging.debug(f"emission matrix shape: {emission.shape}")
    logging.debug(f"transition matrix shape: {transition.shape}")


    ## Saving the files for inference
    ## We're doing this for you :)
    ## TODO: Just Uncomment the following lines when you're ready!
    
    np.savetxt(args.init, init)
    np.savetxt(args.emission, emission)
    np.savetxt(args.transition, transition)

    return 


# No need to change anything beyond this point
if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    main(args)
