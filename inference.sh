# ./inference.sh model_ckpt_path test_data_dir cpu_cores
conda activate jigsaw
python inference.py --model=$1 --test_data=$2 --cpu_cores=$3
# This saves the embeddings in the predicted_embeddings.npy
conda deactivate
conda activate cddd
python get_smiles.py --use_gpu --cpu_threads=$3 --input_file="predicted_embeddings.npy" --output_file="predicted_smiles.txt"
conda deactivate
conda activate jigsaw
python score.py --data_dir=$2

#
# python inference.py --model="checkpoints/checkpoint_11.pt" --test_data="data/test/" --cpu_cores=36
# conda deactivate
# conda activate cddd
# python get_smiles.py --use_gpu --cpu_threads=36 --input_file="predicted_embeddings.npy" --output_file="predicted_smiles.txt"
# conda deactivate
# conda activate jigsaw 
# python score.py --data_dir="data/test/"
