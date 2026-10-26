# FedMed – Weeks 1 & 2

Cross-silo federated 3D brain-tumour segmentation (BraTS / MSD Task01) with Flower + MONAI.

## Layout
| File | Purpose |
|---|---|
| `config.py` | All settings (nodes, ROI, rounds, ports). `FEDMED_SYNTHETIC=1` = tiny fake data |
| `data.py` | BraTS loading, global train/val split, per-hospital shards |
| `model.py` | MONAI 3D U-Net + NumPy weight conversion |
| `engine.py` | Train/eval loops, Dice loss, sliding-window validation (TC/WT/ET Dice) |
| `train_centralized.py` | **Week 1** centralized baseline |
| `client.py` | **Week 1/2** hospital node (+ status port 8081+id) |
| `server.py` | **Week 2** FedAvg aggregator, TLS, per-round JSON history |
| `certs/generate_certs.sh` | **Week 2** private CA + server cert for gRPC TLS |
| `run_federated.sh` | Launch server + 3 hospitals locally |
| `compare.py` | Mid-project audit: federated vs centralized Dice |

## Setup
```bash
pip install -r requirements.txt
```

## 0. Smoke test (no download, minutes on CPU)
```bash
export FEDMED_SYNTHETIC=1
python train_centralized.py --epochs 3
ROUNDS=3 LOCAL_EPOCHS=1 ./run_federated.sh
python compare.py
```

## 1. Real run on BraTS
```bash
unset FEDMED_SYNTHETIC
python download_data.py                  # ~7 GB, once
python train_centralized.py              # Week 1 baseline -> outputs/baseline_metrics.json
./run_federated.sh                       # Week 2 federated  -> outputs/federated_metrics.json
python compare.py                        # mid-project audit
```
Check nodes while running: `curl http://127.0.0.1:8081` (also 8082, 8083).

## Design notes
- **Fair comparison:** baseline epochs default to `rounds x local_epochs`; both use the same
  global validation split (federated val = union of the 3 hospitals' val shards).
- **Privacy boundary:** clients return only weight arrays + scalar metrics. Raw volumes never leave `client.py`.
- **InstanceNorm** (no running stats) keeps FedAvg well-defined.
- **FedAvg** is weighted by each hospital's number of training samples.
- **TLS:** Flower provides server-authenticated TLS (clients verify the server via `ca.crt`). Mutual TLS is not built in.
- Flower clients dial out to one server port, so the "3 ports" are per-node status endpoints.
- Partitioning is IID round-robin; switch to a Dirichlet split in `data.partition` to test non-IID hospitals.
- Optional later: lower `min_fit_clients` to 2 for the Mid-Project "node drops offline" resilience test.
