# NanyinHGNN
# CUDA 11.8
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia 
pytorch_lightning==2.2
 # If you have installed dgl-cuXX package, please uninstall it first. 
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html                                                  
pip install numpy==1.22.4

python scripts/generate_enhanced.py --model 1.ckpt --output generated_music --tempo 100 --temperature 0.8
python train_two_stage.py --config configs/two_stage_training.yaml --batch_size 2 --num_workers 2 --current_stage 1
python train_two_stage.py --config configs/two_stage_training.yaml --batch_size 2 --num_workers 2 --current_stage 2 --resume_from_checkpoint nanyin_model_stage1_20250328_033403.ckpt
