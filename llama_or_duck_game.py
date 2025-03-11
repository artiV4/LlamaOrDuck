import pygame
import sys
import os
import random

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BLUE_GRAY = (64, 64, 64)
RED = (220, 50, 50)

# Font setup
font = pygame.font.Font(None, 50)

# Difficulty Options
options = [0, 15, 30, 60, 120] # 0 is here for debugging, remember to remove
selected_index = 0
selected_option = 0

# Function to calculate the color and width of the progress bar
def get_progress_bar_color(time_left, total_time):
    ratio = time_left / total_time  # Ratio of time left to total time
    red = int(255 * (1 - ratio))   # Red increases as time runs out
    green = int(255 * ratio)       # Green decreases as time runs out
    return red, green, 0          # RGB color (Red, Green, Blue)


def get_balanced_sample(base_path):
    duck_path = os.path.join(base_path, "duck")
    llama_path = os.path.join(base_path, "llama")

    # Get image file names
    duck_images = [os.path.join(duck_path, f) for f in os.listdir(duck_path) if f.endswith(('.jpg', '.png'))]
    llama_images = [os.path.join(llama_path, f) for f in os.listdir(llama_path) if f.endswith(('.jpg', '.png'))]

    # Shuffle images
    random.shuffle(duck_images)
    random.shuffle(llama_images)

    # Determine the max number of pairs we can take
    max_pairs = min(len(duck_images), len(llama_images))
    total_possible_samples = 2 * max_pairs  # Always even to maintain balance
    half_size = total_possible_samples // 2  # Equal split between ducks and llamas

    # Take an equal number of samples from both categories
    final_sample = random.sample(duck_images, half_size) + random.sample(llama_images, half_size)
    random.shuffle(final_sample)  # Shuffle to mix duck and llama images

    return final_sample


def draw_menu():
    screen.fill(BLUE_GRAY)
    title_text = font.render("Select The Number of Seconds:", True, WHITE)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT / 2 - 200))

    for i, option in enumerate(options):
        color = YELLOW if i == selected_index else WHITE
        text = font.render(str(option), True, color)
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT / 2 - (50 + i * -60)))

    pygame.display.flip()


# Set up display dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)  # Makes the window resizable
pygame.display.set_caption('Llama or Duck?')

# Set up game clock
clock = pygame.time.Clock()

dataset_path = 'dataset/data/test'
sample = get_balanced_sample(dataset_path)

# Menu loop
running = True
while running:
    draw_menu()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.VIDEORESIZE:
            WIDTH, HEIGHT = event.w, event.h
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                selected_index = (selected_index + 1) % len(options)
            elif event.key == pygame.K_UP:
                selected_index = (selected_index - 1) % len(options)
            elif event.key == pygame.K_RETURN:
                selected_option = options[selected_index]
                running = False  # Exit menu and start game

# Load and play music
pygame.mixer.music.load("assets/song.mp3")
pygame.mixer.music.play(-1, 0.0)

# Main game loop
running = True
start_time = pygame.time.get_ticks()
image_timer = 0
current_image = random.choice(sample)
while running:
    elapsed_time = (pygame.time.get_ticks() - start_time) / 1000  # Convert to seconds
    remaining_time = max(0, selected_option - int(elapsed_time))
    if elapsed_time >= selected_option:
        running = False  # Exit game loop and move to another menu

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.VIDEORESIZE:
            WIDTH, HEIGHT = event.w, event.h

    # Fill the screen
    screen.fill(BLUE_GRAY)

    if pygame.time.get_ticks() - image_timer > 750: # If the current image has been up for 0.75 seconds
        current_image = random.choice(sample)
        image_timer = pygame.time.get_ticks()

    if current_image:
        img = pygame.image.load(current_image)
        img = pygame.transform.scale(img, (WIDTH, HEIGHT))  # Scale image to fit screen
        screen.blit(img, (0, 0))

    # Draw the shadow first
    shadow_offset = 2  # You can adjust the offset for shadow appearance
    shadow_text = font.render(f"{remaining_time:03}", True, BLUE_GRAY)
    screen.blit(shadow_text, (WIDTH - 80 + shadow_offset, 20 + shadow_offset))  # Offset shadow

    # Draw the timer text on top
    timer_text = font.render(f"{remaining_time:03}", True, WHITE)
    screen.blit(timer_text, (WIDTH - 80, 20))  # Display in top right

    # Calculate the progress bar shrinking effect
    elapsed_since_image = (pygame.time.get_ticks() - image_timer) / 1000  # Time passed since image change
    progress_bar_color = get_progress_bar_color(0.75 - elapsed_since_image, 0.75)

    # Calculate the shrinking parts of the progress bar
    left_width = int((WIDTH / 2) * (elapsed_since_image / 0.75))  # Left side shrinking width
    right_width = int((WIDTH / 2) * (elapsed_since_image / 0.75))  # Right side shrinking width

    # Draw the progress bar
    pygame.draw.rect(screen, progress_bar_color, (left_width, HEIGHT - 10, WIDTH - left_width - right_width, 10))  # Middle part of bar

    # Update the display
    pygame.display.flip()

    # Frame rate (60 FPS)
    clock.tick(60)

pygame.mixer.music.stop()  # Stop music

# Quit Pygame

game_over_text = font.render("Game Over.", True, RED)
statistics_label_text = font.render("Statistics:", True, WHITE)
statistics_text = font.render("", True, WHITE)

# Fill the screen
screen.fill(BLUE_GRAY)
screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT/2 - 250))
screen.blit(statistics_label_text, (WIDTH // 2 - statistics_label_text.get_width() // 2, HEIGHT/2 - 200))

# Menu loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.flip()
pygame.quit()
sys.exit()
