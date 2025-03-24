import pygame
import sys
import os
import random
import csv
import datetime

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Colors
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE_GRAY = (64, 64, 64)
RED = (220, 50, 50)

# Macros
LLAMA = 1
DUCK = 0
TIME_TO_GUESS = 1.0

# Font setup
font = pygame.font.Font(None, 50)
font_small = pygame.font.Font(None, 20)

# Difficulty Options
options = [15, 30, 60, 120] # 0 is here for debugging, remember to remove
selected_index = 0
selected_option = 0

def save_statistics_to_csv(statistics):
    if not os.path.exists("data/"):
        os.mkdir("data/")

    filename = f"data/{selected_option}s_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(["True Label", "User Choice", "Reaction Time (s)"])

        # Write each entry
        for entry in statistics:
            true_label, user_choice, reaction_time = entry
            writer.writerow(["duck" if true_label == DUCK else "llama", "duck" if user_choice == DUCK else "llama" if user_choice == LLAMA else "none", reaction_time])

# Function to calculate the color and width of the progress bar
def get_progress_bar_color(time_left, total_time):
    ratio = time_left / total_time
    red = int(255 * (1 - ratio))
    green = int(255 * ratio)
    return red, green, 0

# Function to calculate the color of the accuracy statistic text
def get_accuracy_color(accuracy):
    accuracy = max(0, min(accuracy, 1))  # Ensure accuracy is within [0, 1]
    if accuracy >= 0.5:
        red = int(510 * (1 - accuracy))
        green = 255
    else:
        red = 255
        green = int(510 * accuracy)
    return red, green, 0

# Function to calculate the color of the average guess time statistic text
def get_average_time_color(average_time):
    ratio = average_time / TIME_TO_GUESS
    red = int(255 * (ratio))
    green = int(255 * 1 - ratio)
    return red, green, 0

def load_image_paths(base_path):
    duck_path = os.path.join(base_path, "animal duck")
    llama_path = os.path.join(base_path, "llama")

    # Get image file names
    duck_images = [os.path.join(duck_path, f) for f in os.listdir(duck_path) if f.endswith(('.jpg', '.png'))]
    llama_images = [os.path.join(llama_path, f) for f in os.listdir(llama_path) if f.endswith(('.jpg', '.png'))]

    # Shuffle images
    random.shuffle(duck_images)
    random.shuffle(llama_images)
    return duck_images, llama_images


def draw_menu():
    screen.fill(BLUE_GRAY)
    title_text = font.render("Select The Number of Seconds:", True, WHITE)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT / 2 - 200))

    for i, option in enumerate(options):
        color = YELLOW if i == selected_index else WHITE
        text = font.render(str(option), True, color)
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT / 2 - (50 + i * -60)))

    pygame.display.flip()

def draw_statistics(statistics):
    # Fill the screen
    screen.fill(BLUE_GRAY)
    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT / 2 - 250))
    screen.blit(statistics_label_text, (WIDTH // 2 - statistics_label_text.get_width() // 2, HEIGHT / 2 - 150))

    # Compute statistics
    total_guesses = len(statistics)
    correct_guesses = sum(1 for entry in statistics if entry[0] == entry[1])
    accuracy = (correct_guesses / total_guesses) * 100 if total_guesses > 0 else 0
    avg_reaction_time = sum(entry[2] for entry in statistics) / total_guesses if total_guesses > 0 else 0
    guess_ratio = f"{correct_guesses}/{total_guesses}"  # Ratio of correct to total guesses

    # Render statistics
    accuracy_text = font_small.render(f"Accuracy: {accuracy:.2f}%", True, get_accuracy_color(accuracy / 100))
    avg_time_text = font_small.render(f"Avg Reaction Time: {avg_reaction_time:.2f}s", True, get_average_time_color(avg_reaction_time))
    total_guesses_text = font_small.render(f"Total Guesses: {total_guesses}", True, WHITE)
    correct_guesses_text = font_small.render(f"Correct Guesses: {correct_guesses} ({guess_ratio})", True, WHITE)

    # Display statistics
    screen.blit(accuracy_text, (WIDTH // 2 - accuracy_text.get_width() // 2, HEIGHT / 2 - 50))
    screen.blit(avg_time_text, (WIDTH // 2 - avg_time_text.get_width() // 2, HEIGHT / 2))
    screen.blit(total_guesses_text, (WIDTH // 2 - total_guesses_text.get_width() // 2, HEIGHT / 2 + 50))
    screen.blit(correct_guesses_text, (WIDTH // 2 - correct_guesses_text.get_width() // 2, HEIGHT / 2 + 100))

    # Update display
    pygame.display.flip()

# Set up display dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)  # Makes the window resizable
pygame.display.set_caption('Llama or Duck?')

# Set up game clock
clock = pygame.time.Clock()

dataset_path = 'dataset/data/test'
image_paths = load_image_paths(dataset_path)


while True:
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

    # Timing variables
    image_timer = pygame.time.get_ticks()

    # Image choice variables
    llama_or_duck = random.choice([LLAMA,DUCK])    # Llama is 0, Duck is 1.
    current_image = random.choice(image_paths[llama_or_duck])

    # Statistic tracking variables
    statistics = []
    user_choice = -1
    elapsed_since_image = 0

    # Main game loop
    running = True
    start_time = pygame.time.get_ticks()
    while running:
        elapsed_time = (pygame.time.get_ticks() - start_time) / 1000  # Convert to seconds
        remaining_time = max(0, selected_option - int(elapsed_time))
        elapsed_since_image = min(TIME_TO_GUESS, (pygame.time.get_ticks() - image_timer) / 1000)  # Time passed since image change
        if elapsed_time >= selected_option:
            running = False  # Exit game loop and move to another menu

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = event.w, event.h
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    user_choice = LLAMA
                    image_timer = pygame.time.get_ticks() - (TIME_TO_GUESS * 1000)
                elif event.key == pygame.K_RIGHT:
                    user_choice = DUCK
                    image_timer = pygame.time.get_ticks() - (TIME_TO_GUESS * 1000)

        # Fill the screen
        screen.fill(BLUE_GRAY)

        if pygame.time.get_ticks() - image_timer >= (TIME_TO_GUESS * 1000): # If the current image has been up for TIME_TO_GUESS seconds
            statistics.append([llama_or_duck, user_choice, elapsed_since_image])
            user_choice = -1
            llama_or_duck = random.choice([LLAMA, DUCK])  # Llama is 1, Duck is 0.
            current_image = random.choice(image_paths[llama_or_duck])
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
        progress_bar_color = get_progress_bar_color(TIME_TO_GUESS - elapsed_since_image, TIME_TO_GUESS)

        # Calculate the shrinking parts of the progress bar
        left_width = int((WIDTH / 2) * elapsed_since_image)  # Left side shrinking width
        right_width = int((WIDTH / 2) * elapsed_since_image)  # Right side shrinking width

        # Draw the progress bar
        pygame.draw.rect(screen, progress_bar_color, (left_width, HEIGHT - 10, WIDTH - left_width - right_width, 10))  # Middle part of bar

        # Update the display
        pygame.display.flip()

        # Frame rate (60 FPS)
        clock.tick(60)

    pygame.mixer.music.stop()  # Stop music

    game_over_text = font.render("Game Over.", True, RED)
    statistics_label_text = font.render("Statistics:", True, WHITE)
    statistics_text = font.render("", True, WHITE)

    draw_statistics(statistics)
    save_statistics_to_csv(statistics)

    # Menu loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    running = False
        if not running:
            break