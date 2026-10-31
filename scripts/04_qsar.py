"""Step 4 - Ligand-based QSAR model (Random Forest on Morgan fingerprints).

  * Morgan fingerprints (radius 2, 2048 bits)
  * scaffold split 80/20 (honest: test molecules have different scaffolds from training)
  * 5-fold scaffold-grouped cross-validation on the training set
  * Y-randomisation (shuffled labels must destroy performance)
  * applicability domain via max Tanimoto similarity to the training set

Usage:  python scripts/04_qsar.py
"""
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_score

from common import FIG, PROC, RES, SEED, save_json

GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def scaffold_split(smiles, frac_train=0.8):
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
        groups[scaf].append(i)
    ordered = sorted(groups.values(), key=lambda idx: (-len(idx), idx[0]))
    n_train = int(frac_train * len(smiles))
    train, test = [], []
    for idx in ordered:
        (train if len(train) + len(idx) <= n_train else test).extend(idx)
    scaf_ids = np.zeros(len(smiles), dtype=int)
    for gid, idx in enumerate(groups.values()):
        scaf_ids[idx] = gid
    return np.array(train), np.array(test), scaf_ids


def bit_fragment(mol, bit):
    """Human-readable substructure behind one Morgan bit (first example)."""
    try:
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        GEN.GetFingerprint(mol, additionalOutput=ao)
        info = ao.GetBitInfoMap()
        if bit not in info:
            return None
        atom, rad = info[bit][0]
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom)
        atoms = {atom}
        for b in env:
            bd = mol.GetBondWithIdx(b)
            atoms.update([bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()])
        return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms),
                                        bondsToUse=list(env), rootedAtAtom=atom)
    except Exception:
        return None


def main():
    df = pd.read_csv(PROC / "chembl_clean.csv")
    mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    fps = [GEN.GetFingerprint(m) for m in mols]
    X = np.array([GEN.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8)
    y = df["pIC50"].values

    tr, te, scaf_ids = scaffold_split(df["smiles"].tolist())
    print(f"Train: {len(tr)}  Test: {len(te)}  (scaffold split)")
    df.iloc[tr].to_csv(PROC / "train.csv", index=False)
    df.iloc[te].to_csv(PROC / "test.csv", index=False)

    rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
    rf.fit(X[tr], y[tr])
    pred = rf.predict(X[te])
    r2 = r2_score(y[te], pred)
    rmse = float(np.sqrt(mean_squared_error(y[te], pred)))

    # 5-fold CV, folds never split a scaffold between train and validation
    cv = GroupKFold(n_splits=5)
    cv_r2 = cross_val_score(RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
                            X[tr], y[tr], groups=scaf_ids[tr], cv=cv, scoring="r2")
    cv_rmse = -cross_val_score(RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
                               X[tr], y[tr], groups=scaf_ids[tr], cv=cv,
                               scoring="neg_root_mean_squared_error")

    # Y-randomisation: shuffle training labels, R2 on the real test set should collapse
    rng = np.random.default_rng(SEED)
    yr_r2 = []
    for _ in range(10):
        m = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
        m.fit(X[tr], rng.permutation(y[tr]))
        yr_r2.append(r2_score(y[te], m.predict(X[te])))

    # Applicability domain: similarity of each test molecule to its nearest training molecule
    train_fps = [fps[i] for i in tr]
    max_sim = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)) for i in te])
    err = np.abs(pred - y[te])
    in_ad = max_sim >= 0.4

    metrics = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "test_R2": float(r2), "test_RMSE": rmse,
        "cv5_R2_mean": float(cv_r2.mean()), "cv5_R2_std": float(cv_r2.std()),
        "cv5_RMSE_mean": float(cv_rmse.mean()), "cv5_RMSE_std": float(cv_rmse.std()),
        "y_randomization_R2_mean": float(np.mean(yr_r2)), "y_randomization_R2_max": float(np.max(yr_r2)),
        "AD_threshold_tanimoto": 0.4, "fraction_test_in_AD": float(in_ad.mean()),
        "MAE_in_AD": float(err[in_ad].mean()) if in_ad.any() else None,
        "MAE_outside_AD": float(err[~in_ad].mean()) if (~in_ad).any() else None,
    }
    save_json(metrics, RES / "qsar_metrics.json")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()})

    out = df.iloc[te].copy()
    out["qsar_pred_pIC50"] = pred
    out["max_tanimoto_to_train"] = max_sim
    out.to_csv(RES / "qsar_test_predictions.csv", index=False)

    # --- figures ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.scatter(y[te], pred, s=14, alpha=0.6, color="#2a9d8f")
    lim = [min(y[te].min(), pred.min()) - 0.3, max(y[te].max(), pred.max()) + 0.3]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Experimental pIC50")
    ax.set_ylabel("Predicted pIC50")
    ax.set_title(f"Scaffold-split test set\nR$^2$ = {r2:.2f}, RMSE = {rmse:.2f}")
    ax = axes[1]
    sc = ax.scatter(max_sim, err, s=14, alpha=0.6, c=y[te], cmap="viridis")
    ax.axvline(0.4, color="#e76f51", ls="--", label="AD threshold (0.4)")
    ax.set_xlabel("Max Tanimoto similarity to training set")
    ax.set_ylabel("|prediction error|")
    ax.set_title("Applicability domain")
    ax.legend(frameon=False)
    fig.colorbar(sc, ax=ax, label="experimental pIC50")
    fig.tight_layout()
    fig.savefig(FIG / "qsar_scatter_AD.png", dpi=200)
    plt.close(fig)

    top = np.argsort(rf.feature_importances_)[::-1][:20]
    frag = []
    for b in top:
        s = None
        for i in np.where(X[:, b] == 1)[0][:20]:
            s = bit_fragment(mols[i], int(b))
            if s:
                break
        frag.append(s)
    imp = pd.DataFrame({"bit": top, "importance": rf.feature_importances_[top], "example_fragment": frag})
    imp.to_csv(RES / "qsar_top20_bits.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    labels = [f"bit {b}" + (f"  {f}" if f else "") for b, f in zip(imp["bit"], imp["example_fragment"])]
    ax.barh(range(20)[::-1], imp["importance"], color="#264653")
    ax.set_yticks(range(20)[::-1])
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title("Top 20 Morgan bits")
    fig.tight_layout()
    fig.savefig(FIG / "qsar_top_bits.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
