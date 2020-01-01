
import pygame
import random
from abc import ABC, abstractmethod
from math import *
import cv2
import numpy as np

# ------------------- #
# --- ARENA CLASS --- #
# ------------------- #


class Arena:

    def __init__(self, height, width, color):

        # Dimensions
        self._height = height
        self._width = width

        # Background color (3-element list)
        self._color = color

        # List of objects
        self._objects = {}

        # List of agents
        self._agents = {}

        # Elapsed time
        self._time = 0

        # Create surface for rendering world as an image
        self.display = pygame.Surface((width, height))

    # Register an agent with the arena
    def add_agent(self, agent, id=1):
        self._agents[id] = agent

    # Register an object with the arena
    # TODO - automatically assign non-repeating IDs
    def add_object(self, obj, id=1):
        self._objects[id] = obj

    # Return the count of objects of the specified type in the arena
    def get_num_objects(self, obj_type=None):

        if obj_type is not None:
            obj_type = ArenaObj

        return sum([1 if isinstance(self._objects[o], obj_type) else 0 for o in self._objects ])

    # Retrieve an agent
    def get_agent_by_id(self, id):
        return self._agents[id]

    # Retrieve an object
    def get_object_by_id(self, id):
        return self._objects[id]

    # Remove all objects and reset the display
    def clear(self):
        self._objects = {}
        self._agents = {}
        self.display = pygame.Surface((self._width, self._height))

    # Draw the arena with all objects and agents
    def update(self):

        # Clear display
        self.display.fill(color=self._color)

        # Draw objects
        for obj_id, obj in self._objects.items():
            obj.draw_self(self.display)

        # Draw agents and update vision
        for agent_id, agent in self._agents.items():
            self.update_agent_vision(agent_id)
            agent.draw_self(self.display)

    # Return the vision vector for an agent
    def update_agent_vision(self, agent_id):

        # Get the agent
        agent = self._agents[agent_id]

        # Starting point for vision rays (center of eyeball)
        x_e = int(agent.x + (agent.radius * cos(agent.orientation)))
        y_e = int(agent.y - (agent.radius * sin(agent.orientation)))

        # Compute minimum and maximum ray angles based on agent's orientation and fov
        theta_min = agent.orientation - (agent.eye_fov / 2)

        # Get the current display as a 2D numpy array
        world_arr = pygame.surfarray.array3d(self.display)

        # Number of radii to check over
        rows = int(np.linalg.norm([self._width, self._height], ord=2)) - agent.eye_radius - 1

        # Start with array of angles
        T = (np.linspace(0.0, 1.0, num=agent.eye_resolution) * agent.eye_fov) + theta_min
        T = np.reshape(T, (1, len(T)))

        # Create array of search radii
        R = np.linspace(agent.eye_radius + 2, agent.eye_radius + rows, num=rows + 1)
        R = np.reshape(R, (len(R), 1))

        # Compute X and Y matrices
        X = x_e + np.dot(R, np.cos(T)).astype(np.int16)
        Y = y_e - np.dot(R, np.sin(T)).astype(np.int16)

        # Make sure values fall in correct range
        X[X < 0] = 0
        X[X > self._width - 1] = self._width - 1
        Y[Y < 0] = 0
        Y[Y > self._height - 1] = self._height - 1

        # Get the colors at each coordinate
        C = world_arr[X, Y, :]

        # Make boundary pixels same color as background
        C[np.logical_or(X == 0, Y == 0)] = self._color
        C[np.logical_or(X == self._width - 1, Y == self._height - 1)] = self._color

        # Figure out first color encountered along each ray
        mask = np.all(C == np.array(self._color), axis=2)
        mask = np.logical_not(mask)
        idx = np.argmax(mask, axis=0)
        idx[idx == 0] = -1
        V = C[idx, np.arange(agent.eye_resolution)]
        V = np.reshape(V, (1, agent.eye_resolution, 3))

        # Update current vision for agent
        agent.vision = np.flip(V, axis=0)
        agent.X = X
        agent.Y = Y

    # Advance the arena state given a set of agent actions for the current time step
    # - Apply each agent action
    # - Consume objects where applicable
    # - Check for collisions during movement
    # - Update the display and agent vision vectors
    def tick(self, agent_actions):

        # Apply all agent actions
        # TODO randomize order in which agents take actions
        for agent_id, action in agent_actions.items():

            agent = self._agents[agent_id]

            # Take the action
            if action == "f":
                self.agent_move_lin(agent, dir=1)

            elif action == "b":
                self.agent_move_lin(agent, dir=-1)

            elif action == "cw":
                self.agent_move_ang(agent, dir=1)

            elif action == "ccw":
                self.agent_move_ang(agent, dir=-1)

            elif action == "l":
                pass

            elif action == "r":
                pass

            # If applicable, reduce agent energy
            if isinstance(agent, EnergyAgent):
                agent.use_energy()

        # Apply all object actions
        consumed = []
        for object_id, obj in self._objects.items():

            # Do object actions
            if isinstance(obj, ActionObj):
                obj.do_action()

            # Update consumables
            if isinstance(obj, ConsumableObj):

                # Check if any EnergyAgents are close enough to eat
                for agent_id, agent in self._agents.items():

                    if isinstance(agent, ConsumerAgent):

                        # Get distance between food object and agent
                        agnt_coords = np.array([agent.x, agent.y])
                        food_coords = np.array([obj.x, obj.y])
                        dist = np.linalg.norm(agnt_coords - food_coords, ord=2)

                        # If they are touching, collect food
                        if dist < agent.radius + obj.radius:
                            obj.do_consume(agent)
                            consumed.append(object_id)

        # Remove all consumed objects
        for obj_id in consumed:
            del self._objects[obj_id]

        # Increment time
        self._time += 1

        # Update display, agent vision
        self.update()

    # Move an agent linearly (if possible)
    def agent_move_lin(self, agent, dir=1):

        # Position agent is trying to move to
        noise = + np.random.normal(0.0, 0.1) * agent.speed_lin
        new_x = (agent.x + agent.speed_lin * cos(agent.orientation) * dir) + noise
        new_y = (agent.y - agent.speed_lin * sin(agent.orientation) * dir) + noise

        # Keep agent inside the arena
        min_dist = agent.radius + agent.eye_radius + 2

        if new_x < min_dist:
            new_x = min_dist
        if new_y < min_dist:
            new_y = min_dist
        if new_x > self._width - min_dist:
            new_x = self._width - min_dist
        if new_y > self._height - min_dist:
            new_y = self._height - min_dist

        # TODO: Check for collisions before applying the movement

        agent.x = new_x
        agent.y = new_y

    # Rotate an agent
    def agent_move_ang(self, agent, dir):
        noise = np.random.normal(0.0, agent.speed_ang / 4)
        agent.orientation += (agent.speed_ang * dir) + noise


# ----------------------------------- #
# --- OBJECT CLASS AND SUBCLASSES --- #
# ----------------------------------- #


class ArenaObj(ABC):

    def __init__(self, x, y, color):

        # Base class
        super().__init__()

        # Basic state
        self.x = x
        self.y = y
        self.color = color

        # Required attributes
        self.is_solid = False
        self.is_consumable = False
        self.has_action = False

    @abstractmethod
    def draw_self(self, surface):
        pass


class ActionObj(ArenaObj):

    def __init__(self, x, y, color):
        super().__init__(x, y, color)

    @abstractmethod
    def do_action(self):
        pass


class ConsumableObj(ArenaObj):

    def __init__(self, x, y, color, radius):
        super().__init__(x, y, color)
        self.radius = radius
        self.consumed = False

    @abstractmethod
    def do_consume(self, consumer):
        pass


class FoodObj(ConsumableObj):

    def __init__(self, x, y, color, energy, radius):

        super().__init__(x, y, color, radius)
        self.energy = energy
        self.is_consumable = True

    def draw_self(self, surface):
        pygame.draw.circle(surface, self.color, [self.x, self.y], self.radius)

    def do_consume(self, consumer):

        # If consumer is an EnergyAgent, transfer energy
        if isinstance(consumer, EnergyAgent):
            consumer.add_energy(self.energy)

        # Mark self as consumed
        self.consumed = True


# ---------------------------------- #
# --- AGENT CLASS AND SUBCLASSES --- #
# ---------------------------------- #

class BaseAgent(ArenaObj):

    def __init__(self, x, y, radius, color, orientation, eye_radius, eye_color,
                 eye_fov, eye_resolution, speed_lin=1, speed_ang=pi/8):

        # Initialize base class
        super().__init__(x, y, color)

        # Position / appearance info for agent
        self.radius = radius
        self.orientation = orientation

        # Vision parameters
        self.eye_radius = eye_radius
        self.eye_color = eye_color
        self.eye_fov = eye_fov
        self.eye_resolution = eye_resolution

        # Movement parameters
        self.speed_lin = speed_lin
        self.speed_ang = speed_ang

        # Current vision vector
        self.vision = None
        self.X = None
        self.Y = None

    def draw_self(self, surface):

        # Draw agent body
        pygame.draw.circle(surface, self.color, [int(self.x), int(self.y)], self.radius)

        # Draw agent eyeball
        x_e = int(self.x + (self.radius * cos(self.orientation)))
        y_e = int(self.y - (self.radius * sin(self.orientation)))
        pygame.draw.circle(surface, self.eye_color, [int(x_e), int(y_e)], self.eye_radius)


class ConsumerAgent(BaseAgent):

    def __init__(self, x, y, radius, color, orientation, eye_radius, eye_color,
                 eye_fov, eye_resolution, speed_lin=1, speed_ang=pi/8):

        super().__init__(x, y, radius, color, orientation, eye_radius, eye_color,
                 eye_fov, eye_resolution, speed_lin=speed_lin, speed_ang=speed_ang)


class EnergyAgent(ConsumerAgent):

    def __init__(self, x, y, radius, color, orientation, eye_radius, eye_color,
                 eye_fov, eye_resolution, energy, speed_lin=1, speed_ang=pi/8):

        # Call BaseAgent constructor
        super().__init__(x, y, radius, color, orientation, eye_radius, eye_color,
                 eye_fov, eye_resolution, speed_lin=speed_lin, speed_ang=speed_ang)

        # Add energy attribute
        self._energy = energy

    def add_energy(self, amt=1):
        self._energy += amt

    def use_energy(self, amt=1):
        self._energy -= amt

    def get_energy(self):
        return self._energy


