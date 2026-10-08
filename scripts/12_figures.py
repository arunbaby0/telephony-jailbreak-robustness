"""Phase 7e: fig1 ASR heatmap, fig2 guardrail recall/FPR, fig3 mechanism plots."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = "/data/artifacts"

def fig1():
    asr = pd.read_csv(f"{ART}/metrics/asr.csv")
    for model in asr["model"].unique():
        sub = asr[asr.model == model]
        piv = sub.pivot_table(index="family", columns="channel", values="asr")
        cond_order = [c for c in ["C0","C1","C2","C3","C4","C5","C6","C7","C8"] if c in piv.columns]
        piv = piv[cond_order]
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(piv.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"ASR heatmap - {model}")
        plt.colorbar(im, ax=ax, label="ASR")
        plt.tight_layout()
        plt.savefig(f"{ART}/figures/fig1_asr_heatmap_{model}.pdf")
        plt.close()

def fig2():
    g = pd.read_csv(f"{ART}/metrics/guardrail.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for gname in g["guardrail"].unique():
        sub = g[g.guardrail == gname].sort_values("channel")
        axes[0].plot(sub["channel"], sub["recall"], marker="o", label=gname)
        axes[1].plot(sub["channel"], sub["fpr"], marker="o", label=gname)
    axes[0].set_title("Recall by channel"); axes[0].legend()
    axes[1].set_title("FPR by channel"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{ART}/figures/fig2_guardrail.pdf")
    plt.close()

def fig3():
    surv = pd.read_csv(f"{ART}/metrics/mechanism_perturbation_survival.csv")
    prosody = pd.read_csv(f"{ART}/metrics/mechanism_prosody_retention.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if len(surv):
        s = surv.groupby("channel")["survival_ratio"].mean()
        axes[0].bar(s.index, s.values)
    axes[0].set_title("Perturbation survival ratio (A5)")
    if len(prosody):
        p = prosody.groupby("channel")["f0_range_retention"].mean()
        axes[1].bar(p.index, p.values)
    axes[1].set_title("Prosody F0-range retention (A1-A3)")
    plt.tight_layout()
    plt.savefig(f"{ART}/figures/fig3_mechanism.pdf")
    plt.close()

def main():
    import os
    os.makedirs(f"{ART}/figures", exist_ok=True)
    fig1(); fig2(); fig3()
    print("FIGURES_DONE")

if __name__ == "__main__":
    main()
