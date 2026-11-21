import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
df=pd.read_csv("results/pooling_all_subjects_long.csv")
rois=["V1","V2","V3","EBA","OFA","FFA","OPA","PPA","MPA"]
cols={"supervised":"C0","simclr":"C1","random":"C7"}
stg=["stem","layer1","layer2","layer3","layer4"]
for g in [1,4]:
    fig,ax=plt.subplots(3,3,figsize=(11,8),sharex=True)
    for a,r in zip(ax.flat,rois):
        for m,c in cols.items():
            d=df[(df.roi==r)&(df.model==m)&(df.blk.isna())&(df.g==g)&(df.stage.isin(stg))]
            p=d.pivot_table(index="stage",columns="sub",values="rho").reindex(stg)
            a.errorbar(range(5),p.mean(1),p.std(1),color=c,marker="o",ms=3,label=m,capsize=2)
        a.set_title(f"{r} (n={p.shape[1]})");a.set_xticks(range(5));a.set_xticklabels(stg,rotation=40,fontsize=7)
    ax[0,0].legend(fontsize=7);fig.suptitle(f"RSA rho by stage, {g}x{g} pooling; mean ± SD over subjects")
    fig.tight_layout();fig.savefig(f"figures/pool_profile_g{g}.png",dpi=200);fig.savefig(f"figures/pool_profile_g{g}.pdf")
# table: mean peak rho (g1/g2/g4 stage-level) per roi x model, with ceiling
st=df[df.blk.isna()&df.g.isin([1,2,4])]
pk=st.groupby(["sub","roi","model"]).rho.max().reset_index()
t=pk.groupby(["roi","model"]).rho.agg(["mean","min","max"]).round(3).unstack("model")
print(t)
ce=df.groupby(["roi"]).ceiling.mean().round(2); print("\nmean ceiling (sqrt SB):\n",ce)
t.to_csv("results/summary_peak_rho.csv")
