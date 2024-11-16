TODO: 


code to run: 


CUDA_VISIBLE_DEVICES=2 python train.py --config_file configs/models/rn101_ep50.yaml --datadir /home/samyakr2/multilabel/data/VOC2007/VOCdevkit/VOC2007/ --dataset_config_file /home/samyakr2/Summer24/linear_layer/DualCoOp/configs/datasets/voc2007.yaml --input_size 448 --lr 0.5 --max_epochs 52 --loss_w 0.03 -pp 0.9 --csc --method_name negativecoop
