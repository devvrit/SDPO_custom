#!/bin/bash

# =============================================================================
# Ray Multi-Node Setup Script for SLURM
#
# This script automatically sets up a Ray cluster across SLURM nodes.
# Usage: source ./scripts/setup_ray_multinode.sh
#
# Requirements:
#   - Must be run within a SLURM job with SLURM_JOB_NODELIST set
#   - N_GPUS_PER_NODE environment variable should be set
# =============================================================================

# Only run Ray setup if in SLURM multi-node environment
if [ -n "$SLURM_JOB_NODELIST" ] && [ -n "$SLURM_NNODES" ] && [ "$SLURM_NNODES" -gt 1 ]; then
    echo "============================================"
    echo "Setting up Ray cluster for $SLURM_NNODES nodes"
    echo "============================================"

    # Get the node names and IPs
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)

    head_node=${nodes_array[0]}
    echo "Head node: $head_node"

    # Get head node IP address
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    # Handle IPv6/IPv4 - if there are multiple IPs, pick IPv4
    if [[ "$head_node_ip" == *" "* ]]; then
        IFS=' ' read -ra ADDR <<<"$head_node_ip"
        if [[ ${#ADDR[0]} -gt 16 ]]; then
            head_node_ip=${ADDR[1]}
        else
            head_node_ip=${ADDR[0]}
        fi
        echo "Multiple IPs detected. Using: $head_node_ip"
    fi

    # Ray configuration
    port=6379
    ip_head=$head_node_ip:$port
    export ip_head

    # Use N_GPUS_PER_NODE if set, otherwise default to SLURM_GPUS_PER_NODE
    GPUS_PER_NODE=${N_GPUS_PER_NODE:-${SLURM_GPUS_PER_NODE:-8}}
    CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-64}

    echo "Ray head address: $ip_head"
    echo "GPUs per node: $GPUS_PER_NODE"
    echo "CPUs per task: $CPUS_PER_TASK"
    echo "============================================"

    # Start Ray on head node
    echo "Starting Ray HEAD on $head_node"
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --num-cpus "$CPUS_PER_TASK" --num-gpus "$GPUS_PER_NODE" --block &

    # Wait for head node to be ready
    sleep 10

    # Start Ray on worker nodes
    worker_num=$((SLURM_NNODES - 1))
    echo "Starting $worker_num worker node(s)..."

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Starting Ray WORKER $i on $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$ip_head" \
            --num-cpus "$CPUS_PER_TASK" --num-gpus "$GPUS_PER_NODE" --block &
        sleep 5
    done

    echo "============================================"
    echo "Ray cluster started successfully!"
    echo "Total nodes: $SLURM_NNODES"
    echo "Total GPUs: $((SLURM_NNODES * GPUS_PER_NODE))"
    echo "============================================"

    # Give Ray a moment to fully initialize
    sleep 5

elif [ -n "$SLURM_JOB_NODELIST" ] && [ "$SLURM_NNODES" -eq 1 ]; then
    echo "Running on single SLURM node - Ray will auto-initialize"
else
    echo "Not running in SLURM environment - Ray will auto-initialize"
fi
