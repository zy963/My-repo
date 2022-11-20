import argparse
import numpy as np
from environment import MountainCar, GridWorld


# NOTE: We highly recommend you to write functions for...
# - converting the state to a numpy array given a sparse representation
# - determining which state to visit next
def dict_to_array(dic, state_space):
    array = np.zeros((1, state_space))
    for key, val in dic.items():
        array[0][key] = val
    return array


def get_action(epsilon, state_array, weights, bias, action_space):
    # 1-EPS optimal
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(0, action_space)
    else:
        action = np.argmax(np.dot(state_array, weights) + bias)
    return action


def update(weights, bias, state_space, action_space, state_array, action, state_prime_array,
           reward, gamma, learning_rate):
    w_a = weights[:, action]
    q = np.dot(state_array, w_a) + bias
    f_reward = np.max(np.dot(state_prime_array, weights) + bias)
    g = np.zeros((state_space, action_space))
    for i in range(len(g)):
        g[i, action] = state_array[0, i]

    weights_update = weights - learning_rate * (q - (reward + gamma * f_reward)) * g
    bias_update = bias - learning_rate * (q - (reward + gamma * f_reward))

    return weights_update, bias_update


def main(args):
    # Command line inputs
    mode = args.mode
    weight_out = args.weight_out
    returns_out = args.returns_out
    episodes = args.episodes
    max_iterations = args.max_iterations
    epsilon = args.epsilon
    gamma = args.gamma
    learning_rate = args.learning_rate
    debug = args.debug

    # We will initialize the environment for you:
    if args.environment == 'mc':
        env = MountainCar(mode=mode, debug=debug)
    else:
        env = GridWorld(mode=mode, debug=debug)

    # TODO: Initialize your weights/bias here
    state_space = env.state_space
    action_space = env.action_space
    weights = np.zeros((state_space, action_space))  # Our shape is |A| x |S|, if this helps.
    bias = 0
    # If you decide to fold in the bias (hint: don't), recall how the bias is
    # defined!

    returns = []  # This is where you will save the return after each episode

    for episode in range(episodes):
        # Reset the environment at the start of each episode
        state_dict = env.reset()
        # `state` now is the initial state
        state_array = dict_to_array(state_dict, state_space)
        total_reward = 0
        iteration = 0
        flag = True

        while flag and iteration < max_iterations:
            action = get_action(epsilon, state_array, weights, bias, action_space)
            state_prime_dict, reward, done = env.step(action)

            state_prime_array = dict_to_array(state_prime_dict, state_space)
            total_reward += reward
            weights, bias = update(weights, bias, state_space, action_space, state_array, action, state_prime_array,
                                       reward, gamma, learning_rate)
            # next state
            state_array = state_prime_array
            iteration += 1
            # termination
            if iteration == max_iterations or done:
                flag = False

        returns.append(total_reward)

            # TODO: Fill in what we have to do every iteration
            # Hint 1: `env.step(ACTION)` makes the agent take an action
            #         corresponding to `ACTION` (MUST be an INTEGER)
            # Hint 2: The size of the action space is `env.action_space`, and
            #         the size of the state space is `env.state_space`
            # Hint 3: `ACTION` should be one of 0, 1, ..., env.action_space - 1
            # Hint 4: For Grid World, the action mapping is
            #         {"up": 0, "down": 1, "left": 2, "right": 3}
            #         Remember when you call `env.step()` you have to pass
            #         the INTEGER representing each action!
            # pass  # You can delete this `pass`
    # TODO: Save output files
    with open(returns_out, 'w') as f:
        for i in range(len(returns)):
            f.write('{}\n'.format(returns[i]))

    with open(weight_out, 'w') as f:
        f.write('{}\n'.format(float(bias)))
        for i in range(len(weights)):
            for j in range(len(weights[0])):
                f.write('{}\n'.format(weights[i][j]))


if __name__ == "__main__":
    # No need to change anything here
    parser = argparse.ArgumentParser()
    parser.add_argument('environment', type=str, choices=['mc', 'gw'],
                        help='the environment to use')
    parser.add_argument('mode', type=str, choices=['raw', 'tile'],
                        help='mode to run the environment in')
    parser.add_argument('weight_out', type=str,
                        help='path to output the weights of the linear model')
    parser.add_argument('returns_out', type=str,
                        help='path to output the returns of the agent')
    parser.add_argument('episodes', type=int,
                        help='the number of episodes to train the agent for')
    parser.add_argument('max_iterations', type=int,
                        help='the maximum of the length of an episode')
    parser.add_argument('epsilon', type=float,
                        help='the value of epsilon for epsilon-greedy')
    parser.add_argument('gamma', type=float,
                        help='the discount factor gamma')
    parser.add_argument('learning_rate', type=float,
                        help='the learning rate alpha')
    parser.add_argument('--debug', type=bool, default=False,
                        help='set to True to show logging')
    main(parser.parse_args())


# mc tile mc_tile_weight.out mc_tile_returns.out 25 200 0.0 0.99 0.005
# mc raw mc_raw_weight.out mc_raw_returns.out 4 200 0.05 0.99 0.01
