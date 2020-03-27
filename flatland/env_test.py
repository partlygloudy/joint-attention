
from flatland.flatland_tasks import *
from flatland.flatland_objects import *

import pygame

if __name__ == "__main__":

    test_env = TaskBasicChoice(resolution=64, fov=(3.14))

    pygame.init()

    display_width = 200
    display_height = 225

    game = pygame.display.set_mode((display_width, display_height))
    game_clock = pygame.time.Clock()

    done = False
    while not done:

        # Show next frame
        frame = test_env.render(mode="return").swapaxes(0,1)
        game.blit(pygame.surfarray.make_surface(frame), (0, 0))
        pygame.display.update()

        # Wait for next kepress
        for event in pygame.event.get():

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_w:
                    s_prime, r, done = test_env.step(0)
                elif event.key == pygame.K_s:
                    s_prime, r, done = test_env.step(1)
                elif event.key == pygame.K_a:
                    s_prime, r, done = test_env.step(4)
                elif event.key == pygame.K_d:
                    s_prime, r, done = test_env.step(5)
                elif event.key == pygame.K_q:
                    s_prime, r, done = test_env.step(3)
                elif event.key == pygame.K_e:
                    s_prime, r, done = test_env.step(2)

        game_clock.tick(10)







