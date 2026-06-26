#!/usr/bin/env bash
# EvDehaze on SOTS (RESIDE indoor): event-conditioned ResShift latent diffusion, multi-GPU DDP.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train_sots.sh
#
# Place the training/eval data under datasets/ITS_v2/{hazy,clear,clear_events_preprocessed}
# and datasets/SOTS/nyuhaze500/{hazy,gt,gt_events_preprocessed}, and the VQ-f4 VAE under
# weights/autoencoder_vq_f4.pth.
set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
# single-node multi-GPU NCCL bootstrap over loopback when no eth0 is present
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
CFG=configs/evdehaze_sots.yaml
SAVE=outputs/evdehaze-sots

NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "Launching on $NGPU GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -m torch.distributed.run \
    --standalone --nproc_per_node="$NGPU" \
    main.py --cfg_path "$CFG" --save_dir "$SAVE" "$@"
