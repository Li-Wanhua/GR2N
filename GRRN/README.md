Here we present the codes of GR2N.

------------------------------------------
SR_train.py: training code

SRDataset.py: Dataset API for social relation datasets, whcih organizes the data as image-based.

resnet_roi.py: backbone network with an ROI pooling layer.

GRRN.py: the core code of our proposed GR2N.

RIG.py: The overall framework code, which integrates the codes of resnet_roi.py and GRRN.py.

AverageMeter.py: a class of calculating moving average.

utils/*: some useful functions.

run.sh: program script.
