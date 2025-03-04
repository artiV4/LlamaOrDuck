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
