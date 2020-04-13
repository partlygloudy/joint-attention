from flatland import flatland_tasks
from learning_agents import vpg_rtg
import numpy as np

# -- EXPERIMENT PARAMETERS -- #

# Training algorithm
batch_size = 5000
epochs = 500
policy_lr = 0.5e-4

# Environment params
vision_pixels = 64
vision_fov = 3.1416


# --- TRAINING LOOP --- #

# Create environment
env = flatland_tasks.TaskBasicChoice(resolution=vision_pixels, fov=vision_fov)

# Initialize agent
agent = vpg_rtg.VpgRtgAgent(pixels=vision_pixels, channels=3, num_actions=6, policy_lr=policy_lr)

for epoch in range(epochs):

    # Vars that reset for each batch
    batch_obs = []
    batch_acts = []
    batch_weights = []
    batch_rets = []
    batch_lens = []

    # Vars that reset for each episode
    obs = env.reset()
    done = False
    ep_rew = []

    # Render first episode of each epoch
    rendered_this_epoch = False

    # Continue until batch complete
    while True:

        # Render if we're supposed to
        if not rendered_this_epoch:
            env.render()

        # Record new observation
        batch_obs += (obs.tolist())

        # Choose a new action using policy network
        act = agent.get_action(obs.astype(np.float32))
        if act == 1:
            act = 0   # Convert backwards movement to forwards for now
        obs, rew, done = env.step(act)

        # Record action and reward
        batch_acts.append(act)
        ep_rew.append(rew)

        # Check if episode complete
        if done:

            # Record data from the episode
            ep_ret = sum(ep_rew)
            ep_len = len(ep_rew)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            batch_weights += vpg_rtg.compute_ret_2_go(ep_rew)

            # Reset the game
            obs = env.reset()
            done = False
            ep_rew = []

            # Don't render again this epoch
            rendered_this_epoch = True

            # Check if batch is finished
            if len(batch_obs) > batch_size:
                break

    # Perform training step with data from this batch
    agent.train(batch_obs, batch_acts, batch_weights)

    # Print performance info
    info_str = "Epoch " + str(epoch) + " Complete:\t"
    info_str += "Avg. Return: " + str(sum(batch_rets) / len(batch_rets))
    info_str += "\tEpisodes: " + str(len(batch_rets))
    print(info_str)


