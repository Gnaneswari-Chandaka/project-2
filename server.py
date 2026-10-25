"""WEEK 2 - Central aggregator: broadcasts weights, waits for local training, runs FedAvg.

The server never touches a scan: it only sees weight arrays and scalar metrics.
"""
import argparse
import json

import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

import config as cfg
from model import build_model, get_weights, set_weights


def weighted_avg(metrics):
    """Sample-weighted mean of the scalar metrics reported by the hospitals."""
    total = sum(n for n, _ in metrics)
    return {k: sum(n * m[k] for n, m in metrics) / total for k in metrics[0][1]}


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg + per-round history (JSON, consumed by the Week-4 dashboard) + checkpoints."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.history = {}
        self._model = build_model()

    def _dump(self):
        rounds = [self.history[r] for r in sorted(self.history)]
        (cfg.OUT_DIR / "federated_metrics.json").write_text(json.dumps(rounds, indent=2))

    def aggregate_fit(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        if params is not None:
            set_weights(self._model, parameters_to_ndarrays(params))
            torch.save(self._model.state_dict(), cfg.OUT_DIR / "global_latest.pt")
        rec = self.history.setdefault(server_round, {"round": server_round})
        rec.update(train_loss=metrics.get("train_loss"), fit_clients=len(results), fit_failures=len(failures))
        self._dump()
        return params, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        rec = self.history.setdefault(server_round, {"round": server_round})
        rec.update({"val_loss": loss, **{f"val_{k}": v for k, v in metrics.items()}})
        self._dump()
        print(f"[round {server_round}] {json.dumps(rec)}")
        return loss, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=cfg.NUM_ROUNDS)
    ap.add_argument("--local-epochs", type=int, default=cfg.LOCAL_EPOCHS)
    ap.add_argument("--lr", type=float, default=cfg.LR)
    ap.add_argument("--address", default="0.0.0.0:" + cfg.SERVER_ADDRESS.split(":")[1])
    ap.add_argument("--insecure", action="store_true", help="disable TLS (debugging only)")
    args = ap.parse_args()

    torch.manual_seed(cfg.SEED)
    init = ndarrays_to_parameters(get_weights(build_model()))  # the untrained model sent to hospitals

    strategy = LoggingFedAvg(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=cfg.NUM_NODES, min_evaluate_clients=cfg.NUM_NODES,
        min_available_clients=cfg.NUM_NODES,
        initial_parameters=init,
        on_fit_config_fn=lambda r: {"server_round": r, "local_epochs": args.local_epochs, "lr": args.lr},
        fit_metrics_aggregation_fn=weighted_avg,
        evaluate_metrics_aggregation_fn=weighted_avg,
    )

    certs = None if args.insecure else (
        (cfg.CERT_DIR / "ca.crt").read_bytes(),
        (cfg.CERT_DIR / "server.pem").read_bytes(),
        (cfg.CERT_DIR / "server.key").read_bytes(),
    )
    fl.server.start_server(server_address=args.address,
                           config=fl.server.ServerConfig(num_rounds=args.rounds),
                           strategy=strategy, certificates=certs)


if __name__ == "__main__":
    main()
