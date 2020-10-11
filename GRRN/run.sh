#!/bin/sh


CUDA_VISIBLE_DEVICES=2 python SR_train.py --batch-size=32 \
--test-batch-size=16 --max-person=5 --image-size=224 \
--max-epochs=10 --lr=0.00001 --fc-lr=0.00001 \
--save-model='./Relation_Model/' \
--images-root='/home/disk0/Social_relation/PIPA/image/' \
--train-file-pre='./relation_split/relation_train' \
--valid-file-pre='./relation_split/relation_valid' \
--test-file-pre='./relation_split/relation_test' \
--num-workers=2  --num-classes=16 --time-steps=1 \
--manualSeed=-1