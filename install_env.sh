

source activate base

conda create -n text_arena python=3.10 -y

conda activate text_arena

pip install -e .

pip install -r requirements.txt