Our implementation is built on the official implementation of  [DualCoOp](https://github.com/sunxm2357/DualCoOp).


## Environment

We use Pytorch with python 3.9. 

Use `conda env create -f environment.yml` to create the conda environment.
In the conda environment, install `pycocotools` and `randaugment` with pip:
```
pip install pycocotools
pip install randaugment
```
And follow [the link](https://github.com/KaiyangZhou/Dassl.pytorch) to install `dassl`.


## Training 
### MLR with Partial Labels
Use the following code to learn a model for MLR with Partial Labels
```
python train.py  --config_file configs/models/rn101_ep50.yaml \
--datadir <your_dataset_path> --dataset_config_file configs/datasets/<dataset>.yaml \
--input_size 448 --lr <lr_value>   --loss_w <loss_weight> \
-pp <porition_of_avail_label> --csc --method_name
```
Some Args:
- `method_name` : positivecoop/ negativecoop/ baseline
- `dataset_config_file`: currently the code supports `configs/datasets/coco.yaml` and `configs/datasets/voc2007.yaml`  
- `lr`: `0.001` for VOC2007 and `0.002` for MS-COCO.
- `pp`: from 0 to 1. It specifies the portion of labels are available during the training.
- `loss_w`: to balance the loss scale with different `pp`. We use larger `loss_w` for smaller `pp`.
- `csc`: specify if you want to use class-specific prompts. We suggest to use class-agnostic prompts when `pp` is very small.   
Please refer to `opts.py` for the full argument list.
For Example:
```
python train.py --config_file configs/models/rn101_ep50.yaml --datadir /home/samyakr2/multilabel/data/VOC2007/VOCdevkit/VOC2007/ \
--dataset_config_file /home/samyakr2/Summer24/linear_layer/DualCoOp/configs/datasets/voc2007.yaml \
--input_size 448 --lr 0.5 --max_epochs 52 --loss_w 0.03 -pp 0.9 --csc --method_name negativecoop

```


## Evaluation / Inference
### MLR with Partial Labels
```
python val.py --config_file configs/models/rn101_ep50.yaml \
--datadir <your_dataset_path> --dataset_config_file configs/datasets/<dataset>>.yaml \
--input_size 224  --pretrained <ckpt_path> --csc --method_name
```

Welcome to cite our work if you find it is helpful to your research.
```
@article{rawlekar2024rethinking,
  title={Rethinking Prompting Strategies for Multi-Label Recognition with Partial Annotations},
  author={Rawlekar, Samyak and Bhatnagar, Shubhang and Ahuja, Narendra},
  journal={arXiv preprint arXiv:2409.08381},
  year={2024}
}
```
