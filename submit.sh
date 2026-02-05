#!/bin/bash
#SBATCH --job-name=sdpo
#SBATCH --partition=mi2508x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/work1/agrawal/devvrit/SDPO_custom/logs/sdpo_%j.log
#SBATCH --error=/work1/agrawal/devvrit/SDPO_custom/logs/sdpo_%j.err

# =============================================================================
# Multi-Node SDPO Training Script for AMD MI250 GPUs
# Usage: sbatch submit.sh <script_name>
# Example: sbatch submit.sh tooluse
# =============================================================================

set -e

# Check if script name was provided
if [ -z "$1" ]; then
    echo "Error: No script name provided"
    echo "Usage: sbatch submit.sh <script_name>"
    echo "Available scripts: tooluse, lcb, physics"
    exit 1
fi

SCRIPT_NAME=$1

# Validate script exists
if [ ! -f "$WORK/SDPO_custom/run/${SCRIPT_NAME}.sh" ]; then
    echo "Error: Script run/${SCRIPT_NAME}.sh does not exist"
    echo "Available scripts:"
    ls -1 $WORK/SDPO_custom/run/*.sh 2>/dev/null | xargs -n 1 basename | sed 's/\.sh$//' || echo "None found"
    exit 1
fi

# Create logs directory
mkdir -p $WORK/SDPO_custom/logs

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Nodes: $SLURM_NNODES"
echo "Script: run/${SCRIPT_NAME}.sh"
echo "============================================"

# Change to the working directory
cd $WORK/SDPO_custom

# =============================================================================
# RAY CLUSTER SETUP (must happen outside container)
# =============================================================================

if [ "$SLURM_NNODES" -gt 1 ]; then
    echo "============================================"
    echo "Setting up Ray cluster for $SLURM_NNODES nodes"
    echo "============================================"

    # Get node names and IPs
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)

    head_node=${nodes_array[0]}
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

    # Start Ray on head node (inside container)
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
fi

# =============================================================================
# RUN TRAINING
# =============================================================================

# Run the script via container using srun (--overlap allows running alongside Ray processes)
srun --overlap --ntasks-per-node=1 \
    ./container_exec.sh \
        bash -c "export RAY_ADDRESS=${RAY_ADDRESS:-}; ./run/${SCRIPT_NAME}.sh"

echo "============================================"
echo "Training complete!"
echo "============================================"
