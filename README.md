# img2mol


Since we are using the CDDD network, to run the code successfully the cddd environment needs to be setup
from https://github.com/jrwnter/cddd.


Place the data that you want  to train on in the data folder. The formatting for that is given in dataset.py. To get the embeddings run get_embeddings.py with appropriate arguments.


To  run inference:

Download the pretrained model from the link below and use it to run the following command line arguments.

```
python inference.py --model="checkpoints/model.pt" --test_data="data/test/" --cpu_cores=36

conda deactivate
conda activate cddd
python get_smiles.py --use_gpu --cpu_threads=36 --input_file="predicted_embeddings.npy" --output_file="predicted_smiles.txt"
conda deactivate
conda activate jigsaw 
python score.py --data_dir="data/test/"
```

The training data and the pretrained models are present at https://drive.google.com/drive/folders/1hKyv0wtFCxvcEk3IuLe4IF7iWO9Nj-Ve?usp=sharing.
