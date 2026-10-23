"""Mid-project audit helper: federated vs centralized Dice."""
import json

import config as cfg

base = json.loads((cfg.OUT_DIR / "baseline_metrics.json").read_text())
fed = json.loads((cfg.OUT_DIR / "federated_metrics.json").read_text())
fed_scores = [r["val_dice_mean"] for r in fed if "val_dice_mean" in r]

b, f = base["best"]["dice_mean"], fed_scores[-1]
print(f"Centralized best mean Dice : {b:.4f} (epoch {base['best']['epoch']})")
print(f"Federated final mean Dice  : {f:.4f} (round {len(fed_scores)})")
print(f"Gap                        : {b - f:+.4f}  ({f / b:.1%} of baseline)")
print("Per-round federated Dice   :", " ".join(f"{s:.3f}" for s in fed_scores))
