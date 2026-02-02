# Data Generation

## Setup

Add repository directory to `PYTHONPATH` env variable:

```
export PYTHONPATH=/users/$USER/SDPO:$PYTHONPATH
```

## Benchmark datasets

* LiveCodeBench v6: `python data/load_dataset.py --dataset_name livecodebench/code_generation_lite-v6 --output_path datasets/lcb_v6.json`

## Math Datasets

* Polaris-Dataset-53K: `python data/load_dataset.py --dataset_name POLARIS-Project/Polaris-Dataset-53K --output_path datasets/polaris_53k.json`
* AIME 2024: `python data/load_dataset.py --dataset_name math-ai/aime24 --output_path datasets/aime24.json`
* AIME 2025: `python data/load_dataset.py --dataset_name math-ai/aime25 --output_path datasets/aime25.json`
* MATH 500: `python data/load_dataset.py --dataset_name math-ai/math500 --output_path datasets/math500.json`
* AMC 2023: `python data/load_dataset.py --dataset_name math-ai/amc23 --output_path datasets/amc23.json`
* GSM8K: `python data/load_dataset.py --dataset_name openai/gsm8k --output_path datasets/gsm8k.json`

## Generalization Experiments

* tooluse: the json files are already provided in `datasets/tooluse/train.json` and `datasets/tooluse/test.json`
* sciknoweval Biology: `python data/load_dataset.py --dataset_name Biology --output_path datasets/sciknoweval/biology/biology.json`
* sciknoweval Chemistry: `python data/load_dataset.py --dataset_name Chemistry --output_path datasets/sciknoweval/chemistry/chemistry.json`
* sciknoweval Material: `python data/load_dataset.py --dataset_name Material --output_path datasets/sciknoweval/material/material.json`
* sciknoweval Physics: `python data/load_dataset.py --dataset_name Physics --output_path datasets/sciknoweval/physics/physics.json`


## LiveCodeBench

Create a train/test split of the tests by running:
```bash
python data/split_tests.py \
    --json_path datasets/lcb_v6.json \
    --output_dir datasets/lcb_v6
```

## SciKnowEval
Create a train/test split of the data by running:
```bash
python data/split_tasks.py \
    --json_path datasets/biology.json \
    --output_dir datasets/sciknoweval/biology

python data/split_tasks.py \
    --json_path datasets/chemistry.json \
    --output_dir datasets/sciknoweval/chemistry

python data/split_tasks.py \
    --json_path datasets/material.json \
    --output_dir datasets/sciknoweval/material

python data/split_tasks.py \
    --json_path datasets/physics.json \
    --output_dir datasets/sciknoweval/physics

python data/split_tasks.py \
    --json_path datasets/polaris_53k.json \
    --output_dir datasets/polaris_53k \
    --test_ratio 0.019 \
    --seed 42

```

## Prepocessing
Our implementation uses the `parquet` format for the data. To preprocess the data, run the following command:
```bash
python data/preprocess.py \
    --data_source DATASET_PATH
```
`DATASET_PATH` should contain the `train.json` and `test.json` files.

## Test-Time Training on LiveCodeBench Questions (TBD)
