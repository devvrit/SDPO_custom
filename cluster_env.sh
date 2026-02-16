#!/bin/bash
# Auto-detect cluster and set cluster-specific defaults.
# Source this file; it exports: CLUSTER_TYPE, CLUSTER_DEFAULT_PARTITION, CLUSTER_USE_CONTAINER, CLUSTER_CONDA_ENV

if sinfo -o "%P" --noheader 2>/dev/null | grep -q "^mi"; then
    export CLUSTER_TYPE="amd"
    export CLUSTER_DEFAULT_PARTITION="mi2508x"
    export CLUSTER_USE_CONTAINER=true
    export CLUSTER_CONDA_ENV=""
else
    export CLUSTER_TYPE="aws"
    export CLUSTER_DEFAULT_PARTITION="ml.p4de.24xlarge"
    export CLUSTER_USE_CONTAINER=false
    export CLUSTER_CONDA_ENV="sdpo"
    # Clear AMD/ROCm env vars that conflict with NVIDIA/CUDA
    unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
    unset HIP_VISIBLE_DEVICES 2>/dev/null || true
fi
