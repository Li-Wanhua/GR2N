#!/bin/sh


CUDA_VISIBLE_DEVICES=2,3 python SR_train.py --batch-size=32 \
--test-batch-size=16 --max-epochs=40 --lr=0.00002 --fc-lr=0.00002 \
--save-model='../Relation_Model/' \
--images-root='/home/disk0/Social_relation/PIPA/image/' \
--train-data_file='../relation_split/relation_train.txt' \
--valid-data_file='../relation_split/relation_val.txt' \
--test-data_file='../relation_split/relation_test.txt' \
--num-workers=2  --num-classes=16