#!/bin/bash

# =============================================================================
# Dynamic SDPO Training Script
# Called by launch.sh - handles both single and multi-node automatically
# =============================================================================

set -e

# Get parameters from environment (set by launch.sh via --export)
SCRIPT_NAME=${SCRIPT_NAME:-"tooluse"}
NNODES=${NNODES:-$SLURM_NNODES}

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Nodes: $SLURM_NNODES"
echo "Script: run/${SCRIPT_NAME}.sh"
echo "============================================"

# Change to working directory
cd $WORK/SDPO_custom

# =============================================================================
# RAY CLUSTER SETUP (only for multi-node)
# =============================================================================

# Get head node for both single and multi-node
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

if [ "$SLURM_NNODES" -gt 1 ]; then
    echo "============================================"
    echo "Setting up Ray cluster for $SLURM_NNODES nodes"
    echo "============================================"
    echo "Head node: $head_node"

    # Get head node IP - prefer IPv4 for Ray compatibility
    # EXCLUDE: 127.x.x.x (loopback), 169.254.x.x (link-local)
    head_node_ip=""

    # Method 1: Try to get IPv4 using ip command, exclude loopback and link-local
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
        bash -c "ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '^127\.' | grep -v '^169\.254\.' | head -n1" 2>/dev/null)

    # Method 2: Fallback to hostname and filter, exclude loopback and link-local
    if [ -z "$head_node_ip" ]; then
        all_ips=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --all-ip-addresses)
        echo "All IPs: $all_ips"
        head_node_ip=$(echo "$all_ips" | grep -oE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' | grep -v '^127\.' | grep -v '^169\.254\.' | head -n1)
    fi

    # Method 3: Try getting IP from eth0 or ib0 interface specifically
    if [ -z "$head_node_ip" ]; then
        head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
            bash -c "ip -4 addr show eth0 2>/dev/null || ip -4 addr show ib0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n1" 2>/dev/null)
    fi

    # Method 4: Last resort - hostname -i and clean it
    if [ -z "$head_node_ip" ]; then
        head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -i | awk '{print $1}')
        head_node_ip=${head_node_ip%%%*}
    fi

    echo "Selected IP: $head_node_ip"

    # Detect network interface for PyTorch Gloo networking
    # This prevents Gloo from using localhost (127.0.0.1)
    network_interface=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
        bash -c "ip -4 addr show | grep -B2 '$head_node_ip' | head -n1 | awk '{print \$2}' | sed 's/:$//' " 2>/dev/null)

    if [ -z "$network_interface" ]; then
        # Fallback: try common interfaces
        network_interface=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
            bash -c "ip link show | grep -E '^[0-9]+: (eth|ib|ens|enp)' | head -n1 | awk -F': ' '{print \$2}' " 2>/dev/null)
    fi

    if [ -n "$network_interface" ]; then
        export GLOO_SOCKET_IFNAME=$network_interface
        echo "Network interface for Gloo: $network_interface"
    else
        echo "Warning: Could not detect network interface, Gloo may use localhost"
    fi

    # Ray configuration
    port=6379
    ip_head=$head_node_ip:$port
    export ip_head
    export RAY_ADDRESS=$ip_head

    GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-8}
    CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-64}

    echo "Ray head address: $ip_head"
    echo "GPUs per node: $GPUS_PER_NODE"
    echo "CPUs per task: $CPUS_PER_TASK"
    echo "============================================"

    # Start Ray on head node
    echo "Starting Ray HEAD on $head_node"
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        ./container_exec.sh \
            ray start --head --node-ip-address="$head_node_ip" --port=$port \
                --dashboard-port=8266 \
                --num-cpus "$CPUS_PER_TASK" --num-gpus "$GPUS_PER_NODE" --block &

    sleep 10

    # Start Ray on worker nodes
    worker_num=$((SLURM_NNODES - 1))
    echo "Starting $worker_num worker node(s)..."

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Starting Ray WORKER $i on $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            ./container_exec.sh \
                ray start --address "$ip_head" \
                    --num-cpus "$CPUS_PER_TASK" --num-gpus "$GPUS_PER_NODE" --block &
        sleep 5
    done

    echo "============================================"
    echo "Ray cluster started successfully!"
    echo "Total nodes: $SLURM_NNODES"
    echo "Total GPUs: $((SLURM_NNODES * GPUS_PER_NODE))"
    echo "============================================"

    sleep 5

    # Set ray init address for multi-node
    export RAY_INIT_ADDRESS="auto"
else
    echo "Single-node mode - Ray will auto-initialize"
    export RAY_INIT_ADDRESS=""

    # Also detect network interface for single-node to avoid localhost issues
    network_interface=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
        bash -c "ip link show | grep -E '^[0-9]+: (eth|ib|ens|enp)' | head -n1 | awk -F': ' '{print \$2}' " 2>/dev/null)

    if [ -n "$network_interface" ]; then
        export GLOO_SOCKET_IFNAME=$network_interface
        echo "Network interface for Gloo: $network_interface"
    fi
fi

# =============================================================================
# RUN TRAINING
# =============================================================================

# Build additional args based on configuration
export EXTRA_HYDRA_ARGS="trainer.nnodes=$SLURM_NNODES"
if [ -n "$RAY_INIT_ADDRESS" ]; then
    export EXTRA_HYDRA_ARGS="$EXTRA_HYDRA_ARGS +ray_kwargs.ray_init.address=$RAY_INIT_ADDRESS"
fi

# Run the script via container (--overlap allows running alongside Ray processes)
# Run only on head node as per VERL docs - training script is the orchestrator
srun --overlap --nodes=$SLURM_NNODES --ntasks=1 -w "$head_node" \
    ./container_exec.sh \
        bash -c "export RAY_ADDRESS=${RAY_ADDRESS:-}; export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}; export EXTRA_HYDRA_ARGS='${EXTRA_HYDRA_ARGS}'; ./run/${SCRIPT_NAME}.sh"

echo "============================================"
echo "Training complete!"
echo "============================================"
