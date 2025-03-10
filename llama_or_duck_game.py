import pygame
import sys

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Set up display dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)  # Makes the window resizable
pygame.display.set_caption('Llama or Duck?')

# Set up colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set up game clock
clock = pygame.time.Clock()

# Load and play music
pygame.mixer.music.load("assets/song.mp3")
pygame.mixer.music.play(-1, 0.0)


# Main game loop
running = True
fullscreen = False
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # if event.type == pygame.KEYDOWN:

    # Fill the screen with white
    screen.fill(WHITE)

    # TODO: Make game

    # Update the display
    pygame.display.flip()

    # Frame rate (60 FPS)
    clock.tick(60)

# Quit Pygame
pygame.mixer.music.stop() # Stop music
pygame.quit()
sys.exit()
