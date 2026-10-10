# Data preparation

Run the commands below from the repository root after installation.
Raw datasets and generated parquet files are kept outside the source package.

## LiveCodeBench v6

```bash
mkdir -p datasets
python -m data.load_dataset \
  --dataset_name livecodebench/code_generation_lite-v6 \
  --output_path datasets/lcb_v6.json
python -m data.split_tests \
  --json_path datasets/lcb_v6.json \
  --output_dir datasets/lcb_v6
python -m data.preprocess --data_source datasets/lcb_v6
```

The split command partitions each problem's test cases using the inherited
SDPO preprocessing logic.

## SciKnowEval

Example for Chemistry; Biology, Material, and Physics use the same workflow:

```bash
mkdir -p datasets
python -m data.load_dataset \
  --dataset_name Chemistry \
  --output_path datasets/chemistry.json
python -m data.split_tasks \
  --json_path datasets/chemistry.json \
  --output_dir datasets/chemistry \
  --test_ratio 0.1 --seed 42
python -m data.preprocess --data_source datasets/chemistry
```

## Existing SDPO datasets

Place SDPO-formatted `train.json` and `test.json` in a directory and run:

```bash
python -m data.preprocess --data_source /absolute/path/to/dataset
```

This creates `train.parquet` and `test.parquet` for the training launcher.
The JSON format includes `idx`, `prompt`, `answer`, `tests`, `description`,
`kind`, `dataset`, and `elo`; the loading scripts produce this format.
For ToolUse, obtain the prepared JSON files from the
[upstream dataset directory](https://github.com/lasgroup/SDPO/tree/main/datasets/tooluse).
