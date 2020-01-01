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

    def render(self, scale=1):

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

        cv2.imshow("Food Collection Task", frame)
        cv2.waitKey(10)