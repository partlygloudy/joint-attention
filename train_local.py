from flatland import flatland_tasks
from learning_agents import dqn

# Import other requirements
import time
import random

# -- EXPERIMENT PARAMETERS -- #
training_frames = 2000000  # Total steps to train over
mem_capacity = training_frames  # Replay memory size
epoch_size = 10000  # training_frames / 100  # Number of steps in a single epoch
discount = 0.99
batch_size = 32
k = 1  # Number of steps between training batches
trials = 1  # Number of times to run the whole experiment
random_buffer_len = 5000  # Number of initial random actions to take

e_checkpoints = [0, 1200000, 1700000, training_frames]
e_values = [1.0, 0.1, 0.00, 0.00]
e = e_values[0]

# Function for computing value of e at each frame
def calc_e(checkpoints, values, current_frame):
    i = 1

    try:
        while current_frame > checkpoints[i]:
            i += 1
    except:
        i = len(checkpoints) - 1

    check_next = checkpoints[i]
    check_last = checkpoints[i - 1]
    e_next = values[i]
    e_last = values[i - 1]

    m = (e_next - e_last) / (check_next - check_last)
    x = current_frame - check_last
    b = e_last

    return (m * x) + b


# -- DATA COLLECTION -- #

reward_last_100 = [0.0] * 100
q_last_100 = [0.0] * 100
hist_r = []
hist_e = []
hist_q = []
epoch_counter = 1
game_counter = 0

# -- TRAINING LOOP -- #

# Create environment
env = flatland_tasks.TaskBasicChoice(resolution=64, fov=(3.14/2))

# Initialize agent
agent = dqn.DQNAgent(memsize=mem_capacity, gamma=discount, resolution=64)

# Train agent
current_frame = 0
while current_frame < training_frames:

    # Reset the game
    game_reward = 0.0
    game_q = 0.0
    s = env.reset()

    # Run until game ends
    done = False
    while not done:

        # Choose randomly with probability e
        if current_frame >= random_buffer_len:
            e = calc_e(e_checkpoints, e_values, current_frame)
        if random.random() < e or current_frame < random_buffer_len:
            a = random.randint(0, 5)

        # Otherwise choose best action as predicted by agent
        else:
            a, q = agent.choose_action(s)

        # Take the action
        if a == 1:
            a = 0
        s_prime, r, done = env.step(a)

        # Add experience to memory
        agent.memory.add(s, a, r, s_prime)

        # Update network every k timesteps
        if agent.memory.size >= random_buffer_len and current_frame % k == 0:
            agent.learn_from_memory(batch_size)

        # Update current state
        s = s_prime

        # Update tracking info
        game_reward += r
        # game_q += q
        current_frame += 1

    # Game finished, log data
    reward_last_100[game_counter % 100] = game_reward
    # q_last_100[game_counter % 100] = game_q
    game_counter += 1

    if current_frame >= epoch_counter * epoch_size:
        # Print epoch results
        out_str = "Epoch #" + str(epoch_counter)
        out_str += "\tTotal Games:" + str(game_counter)
        out_str += "\tReward avg: " + str(sum(reward_last_100) / 100.0)
        # out_str += "\tQ avg: " + str(sum(q_last_100) / 100.0)
        out_str += "\te: " + str(e)

        # Add data to history arrays
        hist_r += [sum(reward_last_100) / 100.0]
        # hist_q += [sum(q_last_100) / 100.0]
        hist_e += [e]

        print(out_str)

        # Increment epoch counter
        epoch_counter += 1

        # Save model weights
        agent.q_net.save_weights("checkpoints/checkpoint")


# ### TRIAL VIDEO CAPTURE STUFF ###

import cv2

# Number of trials to record
trials = 20

# Create instance of environment for trial
env = flatland_tasks.TaskBasicChoice(resolution=64, fov=(3.14/2))
env.reset()

# Video writer object
sample_frame = env.render(mode="return")
dims = (sample_frame.shape[1], sample_frame.shape[0])
vid = cv2.VideoWriter('test_vid' + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, dims)

for t in range(1, trials + 1):

    # Reset environment
    s = env.reset()

    # Frame counter
    frames = 0

    done = False
    while not done:
        a, q = agent.choose_action(s)
        s_prime, r, done = env.step(a)
        s = s_prime

        # Save rendered game state
        vid.write(env.render(mode="return"))
        frames += 1

    # Print info
    print("Trial #" + str(t) + " recorded\t-\t" + str(frames) + " frames captured")

# Release VideoWriter object
vid.release()


from matplotlib import pyplot as plt

plt.figure(figsize=(6,5))
plt.plot(hist_e)
plt.plot(hist_r)
plt.xlim(0, 100)
plt.ylim(0, 1.0)
plt.xlabel("Epoch (1 Epoch = 5000 frames)")
plt.legend(["Exploration rate", "Last 100 game reward avg."])
plt.savefig("basic_choice_fig.png")
plt.show()