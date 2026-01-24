export CUDA_VISIBLE_DEVICES=0,1,2,3
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_SOCKET_IFNAME=lo
# NCCL_DEBUG=INFO
torchrun --master_port=7788 --nproc_per_node=4 train.py \
     -c configs/d3rdetr/D3RDETR-S-AITOD.yml --seed=0