singularity exec --rocm \
  -B $HOME:$HOME \
  -B $WORK:$WORK \
  --pwd $PWD \
  $WORK/rocm_sdpo.sif \
  ./run_math_debug.sh # was run_local_sdpo.sh