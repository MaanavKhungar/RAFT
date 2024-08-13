evaluate on demo frames
python3 demo.py --model=checkpoints/raft-sintel.pth --path=demo-frames
python3 demo.py --model=models/raft-sintel.pth --path=demo-frames

train
python3 -u train.py --name raft-chairs --stage chairs --validation chairs --gpus 0 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision 
python3 -u train.py --name raft-things --stage things --validation sintel --gpus 0 --num_steps 120000 --batch_size 5 --lr 0.0001 --image_size 400 720 --wdecay 0.0001 --mixed_precision
python3 -u train.py --name raft-sintel --stage sintel --validation sintel  --gpus 0 --num_steps 300 --batch_size 2 --lr 0.0001 --image_size 368 768 --wdecay 0.00001 --gamma=0.85 --mixed_precision
python3 -u train.py --name raft-kitti  --stage kitti --validation kitti  --gpus 0 --num_steps 50000 --batch_size 5 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --mixed_precision


evaluate