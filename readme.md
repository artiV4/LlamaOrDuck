# Llama or Duck
The classification of an image as a llama or a duck has appeared multiple 
times in pop culture and media as a global meme, highlighting the need for
accurate classification and distinction between the two. This study evaluates 
the accuracy of computer vision models in distinguishing between llamas and 
ducks using convolutional neural networks (CNNs). Classification performance 
on a curated dataset is contrasted with human performance data. The models
evaluated are a basic CNN, AlexNet, and DenseNet-121. It was found that all 
models performed the classification faster than human performance, but not 
always more accurately. The best performance was seen from DenseNet-121 with 
an accuracy of 98.36% and average GPU reaction time of 0.027s compared to 
human performance of 91.25% and average reaction speed of 0.547s. This 
highlights the superior efficiency and accuracy of neural networks for 
certain problems compared to human performance and suggests further exploration 
should be done to incorporate them into daily life.

This is the source code repository for our CS4442 final project at Western University.
# Setting up the project
For consistency and to reduce differences between devices, it would be benificial to use a venv. 
More information found here: https://docs.python.org/3/library/venv.html

## Mac/Linux
Make a venv folder (usually .venv in the project directory:
```
python -m venv <path to venv folder>
```
Activate it: 
```
source ./.venv/bin/activate
```
Install the requirements:
```
pip install -r requirements.txt
```

## Windows
Open cmd in the project folder

```
mkdir venv
python -m venv venv\
venv\scripts\activate.bat
pip install -r requirements.txt
```
I have trouble with pip installing cuda properly so if it doesn't compile with CUDA redo:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

When working on the project, if you add more imports update requirements.txt.

# The Project
The main notebook for the project is `llama_or_duck.ipynb`, which contains the code for training,
and evaluating the models. The models we trained are stored as .pt files at https://drive.google.com/drive/folders/1q3U-6G6n45MhoLH2CMm6TYns1SEnMJn5.
# Llama or duck game
In order to measure the performance of our machine learning models, we have
developed a human interface for classifying llamas and ducks. This interface
will allow us to quantify the difference between human ability to classify llamas vs. ducks 
and that of AI. The game uses the same testing dataset as the test set for evaluating the models.
## Running the game
1. Ensure that all requirements are installed from requirements.txt, including `pygame`
2. Ensure that the llamas and ducks datasets are present and in the correct directory structure (Datasets can be found at the links in `dataset_sources.txt`):
```
dataset/
└── data/
    ├── test/
    │   ├── animal duck/
    │   └── llama/
    ├── train/
    │   ├── animal duck/
    │   └── llama/
    └── val/
        ├── animal duck/
        └── llama/
```
3. Run the game using:
```bash
python3 llama_or_duck_game.py
```
## Playing the game
Upon running the game you will be presented with options for the length of time you will
be playing for (in seconds). Each classification activity will have a time limit of one (1) second, displayed as a bar at the bottom
that shrinks as the timer approaches zero (0).
The next activity's timer begins immediately upon completing the previous (Even if you decide faster than in one (1) second).
If you fail to classify the image in one (1) second, that example will be counted as incorrect in your final displayed score.
### Controls
In the menu:
1. **Up and down arrow keys** to select duration
2. **Return (Enter) key** to begin playing. The game begins immediately upon pressing this, so be prepared.

While playing:
1. **Left arrow key** to choose llama
2. **Right arrow key** to choose duck

Statistics menu:
1. **Return (Enter) key** to go back to the initial menu and play again.

## Gameplay Data
### Data collection
As this game was designed to collect user data, this should be disclosed to the user. By playing llama
or duck game, you are agreeing to contributing your data to our research. That being said, all data is 
collected anonymously upon completion of the selected duration (once the statistics menu appears). The 
generated csv files are to be sent to the llama or duck team by the consenting user.

### Data format
The data is stored in the following directory structure:
```
data/
└── {duration}_{date (YYYY-MM-DD)}_{time (HH-MM)}.csv
```
Each `.csv` file contains the following collums:
1. **True Label** $\in$ {llama, duck}
2. **User Choice** $\in$ {llama, duck}
3. **Reaction Time (s)** $\in$ [0.0, 1.0]

From this data many statistics can be calculated, including but not limited to:
1. **Average decision time**
2. **Accuracy**
3. **Frequency of llama-duck mis-classification**
4. **Frequency of duck-llama mis-classification**

# Data Analysis
Two notebooks were used for analyzing the evaluation data from the models, as well
as the comparison of human performance to that of the models:
1. `data_analysis.ipynb` - Generates plots and compares human performance.
2. `Llama_and_Duck_Math.ipynb` Calculates precision, accuracy, recall and F1 score.
