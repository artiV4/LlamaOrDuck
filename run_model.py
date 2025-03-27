import numpy as np
import os
import random
import csv
import datetime
import torch 
import torchvision as tv
import pygame
from PIL import Image
import models

IMG_SIZE = (224,224)
LLAMA = 1
DUCK = 0
TIME_TO_GUESS = 1.0
options = [15, 30, 60, 120]

for_time = False #CHANGE THIS
allowed_time = options[0] #CHANGE THIS WHEN RUNNING
model_name, base_model = models.get_basicCNN() #CHANGE THIS


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

dataset_path = 'dataset/data/test'
image_paths = load_image_paths(dataset_path)

# Ensure CUDA working
device = None
device_str = None
if torch.cuda.is_available():
    print("CUDA working \U0001F911")
    device_str = 'cuda'
    if('4060' in torch.cuda.get_device_name(0)):
        emoji = '\U0001F60D' 
    elif('GTX' in torch.cuda.get_device_name(0)):
           emoji = '\U0001F642'
    elif('TPU' in torch.cuda.get_device_name(0)):
           emoji = '\U0001F47D'
    else: emoji = '\U0001F601'
    print(f"This user has a {torch.cuda.get_device_name(0)} {emoji}") # flex your GPUs here
else:
    print("CUDA not working - running on CPU \U0001F480")
    device_str = 'cpu'
device = torch.device(device_str)

# Load the trained model
def loadModel(file, model):
    #now, move the model to the appropriate device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if device_str=="cpu":
        model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(file))
    model.eval()
    return model

train_dataset_path = 'dataset/data/train'
test_dataset_path = 'dataset/data/test'

def save_statistics_to_csv(statistics):
    if not os.path.exists("data/"):
        os.mkdir("data/")
    filename_mod = allowed_time+"s" if for_time else "all"
    filename = f"data/{model_name}_{filename_mod}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(["True Label", "User Choice", "Reaction Time (s)"])

        # Write each entry
        for entry in statistics:
            true_label, user_choice, reaction_time = entry
            writer.writerow(["duck" if true_label == DUCK else "llama", "duck" if user_choice == DUCK else "llama" if user_choice == LLAMA else "none", reaction_time])



training_transforms = tv.transforms.Compose([tv.transforms.Resize(IMG_SIZE), tv.transforms.ToTensor()])
train_dataset = tv.datasets.ImageFolder(root = train_dataset_path, transform = training_transforms)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=32, shuffle=False)

def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_a_batch

    mean /= total_images_count
    std /= total_images_count

    return mean, std

# don't need to normalize to training mean, std for densenet
# mean, std = get_mean_and_std(train_loader)
# mean = mean[0]
# std = std[0]

test_transforms_densenet = tv.transforms.Compose([
    tv.transforms.Resize(IMG_SIZE),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
])


model = loadModel(f"{model_name}.pt", base_model)

# Statistic tracking variables
statistics = []
pygame.init()


# Main game loop
running = True
start_time = pygame.time.get_ticks()
if(for_time):
    print("starting timed execution\n")
    while running:
        elapsed_time = (pygame.time.get_ticks() - start_time) / 1000  # Convert to seconds
        print(f"\r{elapsed_time}\t/\t{allowed_time} sec", end='')
        if elapsed_time >= allowed_time:
            running = False

        # Image choice variables
        llama_or_duck = random.choice([LLAMA,DUCK])    # Llama is 0, Duck is 1.
        current_image = random.choice(image_paths[llama_or_duck])

        pred_start_time = pygame.time.get_ticks()
        image = Image.open(current_image)
        # image = np.array(Image.Image.load(current_image))
        # image = torch.Tensor(image).to(device)
        image = test_transforms_densenet(image).unsqueeze(0)
        image = image.to(device)

        # Predict classification
        with torch.no_grad():
            output = model(image)
        _, pred = torch.max(output, 1)
        pred_end_time = pygame.time.get_ticks()
        pred_elapsed_time = (pred_end_time-pred_start_time)/1000
        # if(pred_elapsed_time>(TIME_TO_GUESS * 1000)): pred=-1
        statistics.append([llama_or_duck, pred, pred_elapsed_time])
else:
    print("running on all images")
    all_image_paths = image_paths[LLAMA]+image_paths[DUCK]
    for llama_or_duck in [LLAMA, DUCK]:
        current_path = image_paths[llama_or_duck]
        for current_image in current_path:
            pred_start_time = pygame.time.get_ticks()
            image = Image.open(current_image)
            image = test_transforms_densenet(image).unsqueeze(0)
            image = image.to(device)

            # Predict classification
            with torch.no_grad():
                output = model(image)
            _, pred = torch.max(output, 1)
            pred_end_time = pygame.time.get_ticks()
            pred_elapsed_time = (pred_end_time-pred_start_time)/1000
            # if(pred_elapsed_time>(TIME_TO_GUESS * 1000)): pred=-1
            statistics.append([llama_or_duck, pred, pred_elapsed_time])          
print("\nsaving statistics to file")
save_statistics_to_csv(statistics)