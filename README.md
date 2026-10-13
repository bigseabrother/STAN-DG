# STAN-DG

Code and processed datasets for STAN-DG (Spatio-Temporal Attention Network with Dynamic Graphs), a stock price forecasting model combining technical and fundamental features with industry and geographical relations.

Select the model using its internal name: `TimeXer_paper`.

## Installation

Use Python 3.9 and install the project dependencies:

```bash
conda create -n stan-dg python=3.9
conda activate stan-dg
python -m pip install -r requirements.txt
```

GPU execution requires a compatible PyTorch and CUDA installation.

## Project files

| File or directory | Purpose |
| --- | --- |
| `run.py` | Experiment entry point |
| `TimeXer_paper.py` in the model package | STAN-DG implementation |
| `layers/` | Embedding and attention modules |
| `data_provider/` | Data loading and window construction |
| `exp/` | Training and evaluation |
| `utils/` | Metrics and supporting functions |
| `scripts/paper/` | Experiment scripts |
| `data/` | Datasets and relationship tensors |

## Data

Place each dataset's `train.pkl`, `val.pkl`, and `test.pkl` in the corresponding directory:

| Dataset | Stocks | Directory |
| --- | ---: | --- |
| CSI 300 | 204 | `data/dataset_csi300/` |
| CSI 500 | 310 | `data/dataset_csi500/` |
| CSI 1000 | 513 | `data/dataset_csi1000/` |
| Full dataset | 1,027 | `data/dataset_financial2geo/` |

The PKL arrays are arranged as **time × stocks × features** and are already separated into training, validation, and test partitions. The default `stock3geo` loader uses 114 daily-frequency features, 14 financial features, 5 marketization features, and 4 temporal features. Preserve the original feature and stock order.

Place these relationship files directly in `data/`:

```text
assignment_industry_<suffix>.npy
mask_industry_<suffix>.npy
assignment_region_<suffix>.npy
mask_region_<suffix>.npy
```

| Dataset | Industry suffix | Geographical suffix |
| --- | --- | --- |
| CSI 300 | `300` | `300` |
| CSI 500 | `500` | `500` |
| CSI 1000 | `1000` | `1000` |
| Full dataset | `32` | `4` |

For inverse transformation, use the `scaler.pkl` corresponding to the dataset and closing-price feature.

## Training and evaluation

Set data paths and GPU selection in the scripts, and any hardcoded file-loading paths, to match the local installation. From the repository root, run the required dataset script:

```bash
bash scripts/paper/csi300.sh
bash scripts/paper/csi500.sh
bash scripts/paper/csi1000.sh
bash scripts/paper/csi1800.sh
```

Run each command separately. A script may contain multiple experiment commands; inspect its contents before execution.

The supplied configurations use a **64-step input window**, **24-step forecast horizon**, batch size **2**, learning rate **3e-5**, and up to **240 training epochs**. Other settings are defined in the scripts and `run.py`.

Training and evaluation are handled by `exp/exp_long_term_forecasting.py`. Metrics include **MAE, MSE, MAPE, and RMSE**. The evaluation code controls checkpoint and result locations.

## Model experiments

The scripts may also include STAN-DG ablation and sensitivity configurations:

- `ds_flag`: spatial-attention ablation.
- `ct_flag`: temporal-attention ablation.
- `graph_flag`: graph-module ablation.
- `patch_len`: fundamental-feature patch length.
- `knn`: Top-K neighborhood size.


