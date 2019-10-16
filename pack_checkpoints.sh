tar cvf checkpoints.tar \
models/naturally_trained_prefixed_detector \
models/naturally_trained_prefixed_classifier \
models/adv_trained_prefixed_classifier

modeldir=models/cifar10_ovr_Linf_8.0_iter40_lr0.5_bs300/
# find $modeldir -name "*_ckpt_best" | sort | xargs -I% cat %/checkpoint | grep ^model

best=(27000 10000 28000 38000 26000 36000 50000 18000 16000 32000)
for i in `seq 0 9`
do
    tar -v --append --file=checkpoints.tar "$modeldir"class"$i"_ckpt_best/checkpoint
    tar -v --append --file=checkpoints.tar "$modeldir"class"$i"_ckpt_best/checkpoint-${best[$i]}*
done

gzip checkpoints.tar
aws s3 cp --acl public-read checkpoints.tar.gz s3://asymmetrical-adversarial-training/cifar10/

