import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
subs=["sub-01","sub-02","sub-03","sub-06"]
D=[]
for s in subs:
    d=pd.read_csv(f"results/{s}_rsa_pooling_diff.csv"); d["sub"]=s; D.append(d)
D=pd.concat(D)
D=D[~((D["sub"]=="sub-02")&(D.roi=="MPA"))]
D=D[(D.contrast=="supervised_minus_simclr")&(D.block==0)|(D.contrast=="supervised_minus_simclr")&D.stage.isin(["layer2","layer3","layer4","avgpool"])]
D=D[D.grid.isin([1,2,4])&D.stage.isin(["layer2","layer3","layer4","avgpool"])&D.block.isin([0,"0"])|D.stage.eq("avgpool")]
D["sig_pos"]=(D.ci_lo>0); D["sig_neg"]=(D.ci_hi<0)
g=D.groupby(["roi","stage","grid"]).agg(mean_diff=("diff","mean"),n=("diff","count"),
    n_pos=("diff",lambda x:(x>0).sum()),n_ci_pos=("sig_pos","sum"),n_ci_neg=("sig_neg","sum")).round(3).reset_index()
g.to_csv("results/summary_delta_rho.csv",index=False)
print("Delta rho = sup - simclr, grid 2 (n_ci_pos = subjects whose bootstrap CI excludes 0 on the positive side)")
print(g[g.grid==2].pivot_table(index="roi",columns="stage",values=["mean_diff","n_pos","n_ci_pos","n_ci_neg"]).to_string())

# block-level curves, g1, mean over subjects
L=pd.read_csv("results/pooling_all_subjects_long.csv")
B=L[L.blk.notna()&(L.g==1)].copy()
B["blk"]=B.blk.astype(int).astype(str); B["b"]=B.stage.str[-1].astype(int)*10+B.blk.astype(int)
order=sorted(B.b.unique()); lab=[f"L{o//10}.{o%10}" for o in order]
rois=["V1","V2","V3","EBA","OFA","FFA","OPA","PPA","MPA"]
cols={"supervised":"C0","simclr":"C1","random":"C7"}
fig,ax=plt.subplots(3,3,figsize=(12,8),sharex=True)
for a,r in zip(ax.flat,rois):
    for m,c in cols.items():
        p=B[(B.roi==r)&(B.model==m)].pivot_table(index="b",columns="sub",values="rho").reindex(order)
        a.errorbar(range(len(order)),p.mean(1),p.std(1),color=c,marker="o",ms=3,capsize=2,label=m)
    a.set_title(r);a.set_xticks(range(len(order)));a.set_xticklabels(lab,rotation=90,fontsize=6)
ax[0,0].legend(fontsize=7);fig.suptitle("RSA rho per bottleneck block (1x1 pooling); mean ± SD over subjects")
fig.tight_layout();fig.savefig("figures/block_curves_g1.png",dpi=200);fig.savefig("figures/block_curves_g1.pdf")
