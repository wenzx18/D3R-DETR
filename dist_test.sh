export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_SOCKET_IFNAME=lo
# export SAVE_TEST_VISUALIZE_RESULT=False
# export SAVE_INTERMEDIATE_VISUALIZE_RESULT=True
torchrun --master_port=7778 --nproc_per_node=4 train.py -c configs/d3rdetr/D3RDETR-AITOD.yml --test-only -r checkpoints/D3RDETR-S-AITOD-best.pth