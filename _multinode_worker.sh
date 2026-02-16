#!/bin/bash

# =============================================================================
# Multi-Node SDPO Worker Script
# Called by submit_multinode.sh via sbatch — do not run directly.
#
# Sets up a Ray cluster across SLURM nodes, validates it, then runs training.
#
# Expected environment variables (set via sbatch --export):
#   SCRIPT_DIR, RUN_SCRIPT, SUFFIX, EXTRA_HYDRA_ARGS,
#   MODEL_PATH, DATA_PATH, REQUEUE, REQUEUE_CMD
# =============================================================================

set -e

# Determine execution prefix (container on AMD, direct on AWS)
if [ "${CLUSTER_USE_CONTAINER}" = true ]; then
    EXEC_PREFIX="./container_exec.sh"
else
    EXEC_PREFIX=""
fi

# Activate conda env if specified (AWS cluster)
if [ -n "${CLUSTER_CONDA_ENV}" ]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CLUSTER_CONDA_ENV}"
    echo "Activated conda env: ${CLUSTER_CONDA_ENV}"
    # Clear AMD/ROCm env vars that conflict with NVIDIA/CUDA
    unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
    unset HIP_VISIBLE_DEVICES 2>/dev/null || true
fi

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Nodes: $SLURM_NNODES"
echo "Run script: $RUN_SCRIPT"
echo "Suffix: $SUFFIX"
echo "Requeue: $REQUEUE"
echo "============================================"

cd "$SCRIPT_DIR"

# SSL certs for wandb — must be set before Ray daemons start so all
# Ray-spawned actors inherit it.
if [ -f "${SCRIPT_DIR}/cacert.pem" ]; then
    export SSL_CERT_FILE="${SCRIPT_DIR}/cacert.pem"
fi

# =============================================================================
# 1. DETECT HEAD NODE IP
# =============================================================================

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

echo "Head node: $head_node"

# Get head node IP — need a non-loopback, non-link-local IPv4 address.
# Try multiple methods for robustness across different cluster configurations.
head_node_ip=""

# Method 1: ip command, filter out loopback and link-local
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '^127\.' | grep -v '^169\.254\.' | head -n1" 2>/dev/null) || true

# Method 2: hostname --all-ip-addresses
if [ -z "$head_node_ip" ]; then
    all_ips=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --all-ip-addresses 2>/dev/null) || true
    echo "All IPs: $all_ips"
    head_node_ip=$(echo "$all_ips" | grep -oE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' | grep -v '^127\.' | grep -v '^169\.254\.' | head -n1) || true
fi

# Method 3: specific interfaces (eth0, ib0)
if [ -z "$head_node_ip" ]; then
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
        bash -c "ip -4 addr show eth0 2>/dev/null || ip -4 addr show ib0 2>/dev/null" 2>/dev/null \
        | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n1) || true
fi

# Method 4: hostname -i (last resort)
if [ -z "$head_node_ip" ]; then
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -i 2>/dev/null | awk '{print $1}') || true
    head_node_ip=${head_node_ip%%%*}
fi

if [ -z "$head_node_ip" ]; then
    echo "ERROR: Could not determine head node IP address"
    exit 1
fi

echo "Head node IP: $head_node_ip"

# =============================================================================
# 2. DETECT NETWORK INTERFACE FOR GLOO
# =============================================================================

# PyTorch Gloo backend needs to know which interface to use, otherwise it
# may pick loopback (127.0.0.1) and fail on multi-node communication.
network_interface=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "ip -4 addr show | grep -B2 '$head_node_ip' | head -n1 | awk '{print \$2}' | sed 's/:$//' " 2>/dev/null) || true

if [ -z "$network_interface" ]; then
    network_interface=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
        bash -c "ip link show | grep -E '^[0-9]+: (eth|ib|ens|enp)' | head -n1 | awk -F': ' '{print \$2}' " 2>/dev/null) || true
fi

if [ -n "$network_interface" ]; then
    export GLOO_SOCKET_IFNAME=$network_interface
    echo "Gloo interface: $network_interface"
else
    echo "WARNING: Could not detect network interface, Gloo may use localhost"
fi

# =============================================================================
# 3. START RAY CLUSTER
# =============================================================================

port=6379
ip_head=$head_node_ip:$port
export RAY_ADDRESS=$ip_head

GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-8}
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-64}

echo "============================================"
echo "Starting Ray cluster"
echo "  Head address: $ip_head"
echo "  GPUs/node:    $GPUS_PER_NODE"
echo "  CPUs/node:    $CPUS_PER_TASK"
echo "============================================"

# Start Ray HEAD on head node (backgrounded — runs as daemon with --block)
echo "Starting Ray HEAD on $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    $EXEC_PREFIX \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
            --dashboard-port=8266 \
            --num-cpus "$CPUS_PER_TASK" --num-gpus "$GPUS_PER_NODE" --block &

# Wait for head to initialize before connecting workers
sleep 10

# Start Ray WORKERS on remaining nodes
worker_num=$((SLURM_NNODES - 1))
echo "Starting $worker_num worker node(s)..."

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    if [ -z "$node_i" ]; then
        echo "ERROR: Empty node name for worker $i"
        exit 1
    fi
    echo "Starting Ray WORKER $i on $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        $EXEC_PREFIX \
            ray start --address "$ip_head" \
                --num-cpus "$CPUS_PER_TASK" --num-gpus "$GPUS_PER_NODE" --block &
    sleep 5
done

# Wait for all workers to connect
echo "Waiting for Ray cluster to stabilize..."
sleep 10

# =============================================================================
# 4. VALIDATE RAY CLUSTER
# =============================================================================

echo "Validating Ray cluster..."
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    $EXEC_PREFIX \
        python3 -c "
import ray
try:
    ray.init(address='auto')
    nodes = ray.nodes()
    alive = [n for n in nodes if n['Alive']]
    print(f'Ray cluster: {len(alive)} alive node(s) out of {len(nodes)} total')
    for n in nodes:
        status = 'ALIVE' if n['Alive'] else 'DEAD'
        gpus = n['Resources'].get('GPU', 0)
        print(f'  {n[\"NodeManagerHostname\"]}: {status}, {gpus} GPUs')
    if len(alive) < $SLURM_NNODES:
        print(f'ERROR: Expected $SLURM_NNODES nodes but only {len(alive)} alive')
        ray.shutdown()
        exit(1)
    ray.shutdown()
    print('Ray cluster validation passed!')
except Exception as e:
    print(f'ERROR: Ray cluster validation failed: {e}')
    exit(1)
"

echo "============================================"
echo "Ray cluster ready: $SLURM_NNODES node(s), $((SLURM_NNODES * GPUS_PER_NODE)) total GPUs"
echo "============================================"

# =============================================================================
# 5. RUN TRAINING
# =============================================================================

# Build the command to run inside the container.
# Explicitly export env vars through the srun -> container chain to guarantee
# they survive regardless of SLURM/Singularity env forwarding configuration.
TRAIN_CMD="export RAY_ADDRESS=$ip_head; \
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}; \
export EXTRA_HYDRA_ARGS='$EXTRA_HYDRA_ARGS'; \
export MODEL_PATH='$MODEL_PATH'; \
export DATA_PATH='$DATA_PATH'; \
./$RUN_SCRIPT $SUFFIX"

if [ "$REQUEUE" = true ]; then
    # Trap SIGUSR1 sent by SLURM before time limit
    requeue() {
        echo ""
        echo "============================================"
        echo "Time limit approaching — requeuing job..."
        echo "============================================"
        $REQUEUE_CMD
        exit 0
    }
    trap requeue USR1

    # Run in background so the trap can fire while we wait
    srun --overlap --nodes=$SLURM_NNODES --ntasks=1 -w "$head_node" \
        $EXEC_PREFIX bash -c "$TRAIN_CMD" &
    wait $!
else
    srun --overlap --nodes=$SLURM_NNODES --ntasks=1 -w "$head_node" \
        $EXEC_PREFIX bash -c "$TRAIN_CMD"
fi

echo "============================================"
echo "Training complete!"
echo "============================================"
