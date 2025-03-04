For consistency and to reduce differences between our devices, it would be benificial to use a venv. 
More information found here: https://docs.python.org/3/library/venv.html

simply make a venv folder:
	python -m venv <path to venv folder>

Activate it
	follow platform specific instructions on website

in command prompt run 
	pip install -r requirements.txt


If you add more imports update requirements.txt

For some reason torch doesn't automatically install with CUDA support, so if you have a CUDA compatible GPU run
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

For Windows:
open cmd in project folder

mkdir venv
python -m venv venv\
venv\scripts\activate.bat
pip install -r requirements.txt

I have trouble with pip installing cuda properly so if it doesn't compile with CUDA redo:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126