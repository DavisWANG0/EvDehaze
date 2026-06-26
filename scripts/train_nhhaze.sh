#!/usr/bin/env bash
# EvDehaze on NH-HAZE: event-conditioned ResShift latent diffusion, multi-GPU DDP.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_nhhaze.sh
#
# Place the NH-HAZE data under datasets/NH-HAZE/{train,test}/{hazy,clear,*_events_preprocessed}
# and the VQ-f4 VAE under weights/autoencoder_vq_f4.pth.
set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
# single-node multi-GPU NCCL bootstrap over loopback when no eth0 is present
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
CFG=configs/evdehaze_nhhaze.yaml
SAVE=outputs/evdehaze-nhhaze

NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "Launching on $NGPU GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -m torch.distributed.run \
    --standalone --nproc_per_node="$NGPU" \
    main.py --cfg_path "$CFG" --save_dir "$SAVE" "$@"
