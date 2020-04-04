# Import flatland objects
from .flatland_objects import *

# --------------------------------- #
# --- TASK CLASS AND SUBCLASSES --- #
# --------------------------------- #


class Task(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self):
        pass


# ----------------------------------------------------------------------------------------------- #
# TASK: Agent is rewarded for collecting food. Food also provides energy which the agent
#       has a finite supply of. Ends when the agent runs out of energy or all food is collected.
# ACTIONS:
# 0 - move forward
# 1 - move backward
# 2 - turn clockwise
# 3 - turn counterclockwise
# ---------------------------------------------------------------------------------------------- #
class TaskFood200(Task):

    def __init__(self, foodcount=10):

        # Initialize parent class
        super().__init__()

        # Save foodcount attribute
        self.foodcount_init = foodcount
        self.foodcount_prev = foodcount

        # Create arena
        self.arena = Arena(height=200, width=200, color=[255, 255, 255])

        # Call reset to finish initialization
        self.reset()

        # Keep track of total reward
        self.total_reward = 0

    def reset(self):

        # Reset arena
        self.arena.clear()

        # Reset foodcounts
        self.foodcount_prev = self.foodcount_init

        # Add 10 food objects randomly
        for i in range(self.foodcount_init):

            # Set object attributes
            x_pos = random.randint(0, 200)
            y_pos = random.randint(0, 200)
            color_b = random.randint(150, 255)
            color_g = random.randint(0, 150)
            color_r = random.randint(0, 150)
            food = FoodObj(x_pos, y_pos, [color_b, color_g, color_r], 50, 8)

            # Add to the arena
            self.arena.add_object(food, i)

        # Add a single EnergyAgent
        self.arena.add_agent(EnergyAgent(
            x=random.randint(25, 175),
            y=random.randint(25, 175),
            radius=15,
            color=[0, 255, 0],
            orientation=random.random() * 2 * pi,
            eye_radius=6,
            eye_color=[0, 0, 0],
            eye_fov=pi/4,
            eye_resolution=32,
            energy=100,
            speed_lin=2,
            speed_ang=pi/16
        ))

        # Apply update to arena
        self.arena.update()
        # Get vision vector for agent, return
        return self.arena.get_agent_by_id(1).vision

    def step(self, action):

        # Package action into a dictionary
        agent_action_dict = {}

        if action == 0:
            agent_action_dict[1] = "f"
        elif action == 1:
            agent_action_dict[1] = "b"
        elif action == 2:
            agent_action_dict[1] = "cw"
        elif action == 3:
            agent_action_dict[1] = "ccw"

        # Do action
        self.arena.tick(agent_action_dict)

        # Compute reward
        foodcount = self.arena.get_num_objects(FoodObj)
        r = self.foodcount_prev - foodcount
        self.foodcount_prev = foodcount
        self.total_reward += r

        # Get next state
        s_next = self.arena.get_agent_by_id(1).vision

        # Check for game over
        done = self.arena.get_agent_by_id(1).get_energy() <= 0 or foodcount == 0

        # Return reward, next state, stop indicator
        return s_next, r, done

    def render(self, scale=1, mode="display"):

        # Get arena display as numpy array
        frame = pygame.surfarray.array3d(self.arena.display)

        # Add agent's vision cone to the display
        X = self.arena.get_agent_by_id(1).X
        Y = self.arena.get_agent_by_id(1).Y
        frame[X, Y] = frame[X, Y] - 50
        frame[frame < 0] = 0

        # Make vision vector portion of frame
        vision = self.arena.get_agent_by_id(1).vision
        vision = cv2.resize(vision, (200, 15), interpolation=cv2.INTER_NEAREST)

        # Add vision to the bottom of the frame
        frame = np.append(frame, np.zeros(shape=(1, 200, 3), dtype=np.uint8), axis=0)
        frame = np.append(frame, np.multiply(np.ones(shape=(5, 200, 3), dtype=np.uint8), np.array(255)), axis=0)
        frame = np.append(frame, vision, axis=0)
        frame = np.append(frame, np.multiply(np.ones(shape=(5, 200, 3), dtype=np.uint8), np.array(255)), axis=0)

        # Add text overlay showing current energy level
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_1 = "Energy: " + str(self.arena.get_agent_by_id(1).get_energy())
        text_2 = "Total Reward: " + str(self.total_reward)
        frame = cv2.putText(frame, text_1, (5, 10), font, 0.30, color=(0, 0, 255), thickness=1)
        frame = cv2.putText(frame, text_2, (5, 22), font, 0.30, color=(0, 0, 255), thickness=1)

        # Resize
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)

        if mode == "display":
            cv2.imshow("Food Collection Task", frame)
            cv2.waitKey(10)

        elif mode == "return":
            return frame


# ----------------------------------------------------------------------------------------------- #
# TASK: Agent is rewarded for collecting food. Food also provides energy which the agent
#       has a finite supply of. Ends when the agent runs out of energy or all food is collected.
# ACTIONS:
# 0 - move forward
# 1 - move backward
# 2 - turn clockwise
# 3 - turn counterclockwise
# ---------------------------------------------------------------------------------------------- #
class TaskBasicChoice(Task):

    def __init__(self, resolution=32, fov=pi/3):

        # Initialize parent class
        super().__init__()

        # Parameters
        self.resolution=resolution
        self.fov=fov

        # Create arena
        self.arena = Arena(height=200, width=200, color=[255, 255, 255])

        # Call reset to finish initialization
        self.reset()

    def reset(self):

        # Reset arena
        self.arena.clear()

        # Randomly select which side is good / bad food
        self.good = "left" if random.random() > 0.5 else "right"
        reward1 = 1.0 if self.good == "left" else -1.0
        reward2 = 1.0 if self.good == "right" else -1.0

        # Create 2 reward objects
        obj1 = RewardObj(100, 50, [0, 255, 255], reward1, 8)
        obj2 = RewardObj(100, 150, [0, 255, 255], reward2, 8)

        # Add the objects to the arena
        self.arena.add_object(obj1, 1)
        self.arena.add_object(obj2, 2)

        # Set eyeball position, determine proper orientation
        eye_x = 25
        eye_y = 100
        good_x = 100
        good_y = 50 if self.good == "left" else 150
        eye_orientation = atan2(-(good_y - eye_y), good_x - eye_x)

        # Add eyeball and add to the arena
        self.arena.add_object(EyeballObj(
            x=eye_x,
            y=eye_y,
            color=[255,150,150],
            eye_color=[0,0,0],
            radius=15,
            eye_radius=6,
            orientation=eye_orientation
        ), 3)

        # Add a single EnergyAgent
        self.arena.add_agent(EnergyAgent(
            x=175,
            y=100,
            radius=15,
            color=[0, 255, 0],
            orientation= pi,
            eye_radius=6,
            eye_color=[0, 0, 0],
            eye_fov=self.fov,
            eye_resolution=self.resolution,
            energy=100,
            speed_lin=2,
            speed_ang=pi/16
        ))

        # Apply update to arena
        self.arena.update()

        # Get vision vector for agent, return
        return self.arena.get_agent_by_id(1).vision

    def step(self, action):

        # Package action into a dictionary
        agent_action_dict = {}

        if action == 0:
            agent_action_dict[1] = "f"
        elif action == 1:
            agent_action_dict[1] = "b"
        elif action == 2:
            agent_action_dict[1] = "cw"
        elif action == 3:
            agent_action_dict[1] = "ccw"
        elif action == 4:
            agent_action_dict[1] = "l"
        elif action == 5:
            agent_action_dict[1] = "r"

        # Do action
        tick_data = self.arena.tick(agent_action_dict)

        # Compute reward
        done = False
        r = -0.001
        if tick_data["consumed_count"] > 0:
            r = tick_data["reward_collected"]
            done = True

        # Check if agent out of energy
        done = done or self.arena.get_agent_by_id(1).get_energy() <= 0

        # Get next state
        s_next = self.arena.get_agent_by_id(1).vision

        # Return reward, next state, stop indicator
        return s_next, r, done

    def render(self, scale=1, mode="display", wait=10):

        # Get arena display as numpy array
        frame = pygame.surfarray.array3d(self.arena.display)

        # Add agent's vision cone to the display
        X = self.arena.get_agent_by_id(1).X
        Y = self.arena.get_agent_by_id(1).Y
        frame[X, Y] = frame[X, Y] - 50
        frame[frame < 0] = 0

        # Make vision vector portion of frame
        vision = self.arena.get_agent_by_id(1).vision
        vision = cv2.resize(vision, (200, 15), interpolation=cv2.INTER_NEAREST)

        # Add vision to the bottom of the frame
        frame = np.append(frame, np.zeros(shape=(1, 200, 3), dtype=np.uint8), axis=0)
        frame = np.append(frame, np.multiply(np.ones(shape=(5, 200, 3), dtype=np.uint8), np.array(255)), axis=0)
        frame = np.append(frame, vision, axis=0)
        frame = np.append(frame, np.multiply(np.ones(shape=(5, 200, 3), dtype=np.uint8), np.array(255)), axis=0)

        # Add text overlay showing current energy level
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_1 = "Energy: " + str(self.arena.get_agent_by_id(1).get_energy())
        frame = cv2.putText(frame, text_1, (5, 10), font, 0.30, color=(0, 0, 255), thickness=1)
        text_2 = "Target: " + self.good
        frame = cv2.putText(frame, text_2, (5, 23), font, 0.30, color=(0, 0, 255), thickness=1)

        # Resize
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)

        if mode == "display":
            cv2.imshow("Food Collection Task", frame)
            cv2.waitKey(wait)

        elif mode == "return":
            return frame


# ----------------------------------------------------------------------------------------------- #
# TASK: Agent is rewarded for collecting food. Food also provides energy which the agent
#       has a finite supply of. Ends when the agent runs out of energy or all food is collected.
# ACTIONS:
# 0 - move forward
# 1 - move backward
# 2 - turn clockwise
# 3 - turn counterclockwise
# ---------------------------------------------------------------------------------------------- #
class TaskColoredFood(Task):

    def __init__(self, resolution=32, fov=pi/3, color_good=(200, 0, 0), color_bad=(0, 0, 200)):

        # Initialize parent class
        super().__init__()

        # Parameters
        self.resolution=resolution
        self.fov=fov
        self.color_good = color_good
        self.color_bad = color_bad

        # Create arena
        self.arena = Arena(height=200, width=200, color=[255, 255, 255])

        # Call reset to finish initialization
        self.reset()

    def reset(self):

        # Reset arena
        self.arena.clear()

        # Select two random positions in the top half of the world
        obj1_pos = [random.randint(15, 125), random.randint(15, 185)]
        obj2_pos = [random.randint(15, 125), random.randint(15, 185)]

        # Create 2 reward objects
        obj1 = RewardObj(obj1_pos[0], obj1_pos[1], self.color_good, 1.0, 8)
        obj2 = RewardObj(obj2_pos[0], obj2_pos[1], self.color_bad, -1.0, 8)

        # Add the objects to the arena
        self.arena.add_object(obj1, 1)
        self.arena.add_object(obj2, 2)

        # Add a single EnergyAgent
        self.arena.add_agent(EnergyAgent(
            x=175,
            y=100,
            radius=15,
            color=[0, 255, 0],
            orientation=pi,
            eye_radius=6,
            eye_color=[0, 0, 0],
            eye_fov=self.fov,
            eye_resolution=self.resolution,
            energy=200,
            speed_lin=2,
            speed_ang=pi/16
        ))

        # Apply update to arena
        self.arena.update()

        # Get vision vector for agent, return
        return self.arena.get_agent_by_id(1).vision

    def step(self, action):

        # Package action into a dictionary
        agent_action_dict = {}

        if action == 0:
            agent_action_dict[1] = "f"
        elif action == 1:
            agent_action_dict[1] = "b"
        elif action == 2:
            agent_action_dict[1] = "cw"
        elif action == 3:
            agent_action_dict[1] = "ccw"
        elif action == 4:
            agent_action_dict[1] = "l"
        elif action == 5:
            agent_action_dict[1] = "r"

        # Do action
        tick_data = self.arena.tick(agent_action_dict)

        # Compute reward
        done = False
        r = 0
        if tick_data["consumed_count"] > 0:
            r = tick_data["reward_collected"]
            done = True

        # Check if agent out of energy
        done = done or self.arena.get_agent_by_id(1).get_energy() <= 0

        # Get next state
        s_next = self.arena.get_agent_by_id(1).vision

        # Return reward, next state, stop indicator
        return s_next, r, done

    def render(self, scale=1, mode="display", wait=10):

        # Get arena display as numpy array
        frame = pygame.surfarray.array3d(self.arena.display)

        # Add agent's vision cone to the display
        X = self.arena.get_agent_by_id(1).X
        Y = self.arena.get_agent_by_id(1).Y
        frame[X, Y] = frame[X, Y] - 50
        frame[frame < 0] = 0

        # Make vision vector portion of frame
        vision = self.arena.get_agent_by_id(1).vision
        vision = cv2.resize(vision, (200, 15), interpolation=cv2.INTER_NEAREST)

        # Add vision to the bottom of the frame
        frame = np.append(frame, np.zeros(shape=(1, 200, 3), dtype=np.uint8), axis=0)
        frame = np.append(frame, np.multiply(np.ones(shape=(5, 200, 3), dtype=np.uint8), np.array(255)), axis=0)
        frame = np.append(frame, vision, axis=0)
        frame = np.append(frame, np.multiply(np.ones(shape=(5, 200, 3), dtype=np.uint8), np.array(255)), axis=0)

        # Add text overlay showing current energy level
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_1 = "Energy: " + str(self.arena.get_agent_by_id(1).get_energy())
        frame = cv2.putText(frame, text_1, (5, 10), font, 0.30, color=(0, 0, 255), thickness=1)

        # Resize
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)

        if mode == "display":
            cv2.imshow("Food Collection Task", frame)
            cv2.waitKey(wait)

        elif mode == "return":
            return frame

