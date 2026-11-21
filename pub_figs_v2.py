import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.linewidth":0.6,
    "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})
SUP,SIM="#0072B2","#E69F00"; L3="#56B4E9"
CM={"supervised":SUP,"simclr":SIM}; NM={"supervised":"Supervised","simclr":"SimCLR"}
ROIS=["V1","V2","V3","OFA","FFA","EBA","OPA","PPA","MPA"]
STG=["stem","layer1","layer2","layer3","layer4"]; COLS=STG+["avgpool"]
SUBS=["sub-01","sub-02","sub-03","sub-06"]; MK=dict(zip(SUBS,["o","s","^","D"]))
df=pd.read_csv("results/pooling_all_subjects_long.csv")
st=df[df.blk.isna()&df.g.isin([1,2,4])&df.stage.isin(STG)]
per=pd.concat([st.groupby(["sub","roi","model","stage"]).rho.max().reset_index(),
               df[df.stage=="avgpool"][["sub","roi","model","stage","rho"]]],ignore_index=True)
per=per[per.model.isin(CM)]
ceil=df.groupby(["sub","roi"]).ceiling.first().reset_index()
def save(fig,n):
    fig.savefig(f"figures/{n}.pdf",bbox_inches="tight"); fig.savefig(f"figures/{n}.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# ---------- Fig A: layer x ROI heatmap
fig,axs=plt.subplots(1,2,figsize=(6.0,3.4),sharey=True,constrained_layout=True)
for a,m in zip(axs,CM):
    M=per[per.model==m].groupby(["roi","stage"]).rho.mean().unstack().reindex(index=ROIS,columns=COLS)
    im=a.imshow(M.values,cmap="viridis",vmin=0,vmax=0.40,aspect="auto")
    for i in range(len(ROIS)):
        for j in range(len(COLS)):
            v=M.values[i,j]
            if np.isfinite(v): a.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=6,color="w" if v<0.22 else "k")
        pk=int(np.nanargmax(M.values[i,:5])); a.add_patch(Rectangle((pk-.5,i-.5),1,1,fill=False,ec="red",lw=1.3))
    a.set_xticks(range(6)); a.set_xticklabels(COLS,rotation=45,ha="right",rotation_mode="anchor")
    a.set_title(NM[m],color=CM[m],fontweight="bold"); 
    for s in a.spines.values(): s.set_visible(False)
    a.tick_params(length=0)
axs[0].set_yticks(range(len(ROIS))); axs[0].set_yticklabels(ROIS)
cb=fig.colorbar(im,ax=axs,fraction=0.04,pad=0.02); cb.set_label("RSA ρ (Spearman)"); cb.outline.set_visible(False)
save(fig,"pub2_figA_heatmap")

# ---------- Fig B: per-subject layer profiles with ceiling band
fig,axs=plt.subplots(3,3,figsize=(7.2,6.0),sharex=True,constrained_layout=True)
for a,r in zip(axs.flat,ROIS):
    c=ceil[ceil.roi==r].ceiling
    a.axhspan(c.min(),c.max(),color="0.82",lw=0)
    for m in CM:
        p=per[(per.roi==r)&(per.model==m)].pivot_table(index="stage",columns="sub",values="rho").reindex(COLS)
        for s in p.columns: a.plot(range(6),p[s],color=CM[m],lw=0.6,alpha=0.45)
        a.plot(range(6),p.mean(1),color=CM[m],lw=1.8,marker="o",ms=3)
    a.set_title(f"{r} (n={p.shape[1]})",fontsize=8); a.set_ylim(bottom=0)
    a.set_xticks(range(6)); a.set_xticklabels(COLS,rotation=45,ha="right",rotation_mode="anchor")
for a in axs[:,0]: a.set_ylabel("RSA ρ")
h=[Line2D([],[],color=SUP,lw=1.8,marker="o",ms=3,label="Supervised (mean)"),
   Line2D([],[],color=SIM,lw=1.8,marker="o",ms=3,label="SimCLR (mean)"),
   Line2D([],[],color="0.4",lw=0.6,alpha=0.6,label="Individual subjects"),
   Patch(color="0.82",label="Noise ceiling range (√SB, across subjects)")]
fig.legend(handles=h,loc="upper center",bbox_to_anchor=(0.5,0.0),ncol=4,frameon=False,fontsize=7)
save(fig,"pub2_figB_profiles")

# ---------- Fig E: supervised - SimCLR at layer2 / layer3 (2x2 pooling)
D=[]
for s in SUBS:
    d=pd.read_csv(f"results/{s}_rsa_pooling_diff.csv"); d["sub"]=s; D.append(d)
D=pd.concat(D); D=D[~((D["sub"]=="sub-02")&(D.roi=="MPA"))]
D=D[(D.contrast=="supervised_minus_simclr")&(D.grid==2)&(D.block==0)&D.stage.isin(["layer2","layer3"])]
fig,a=plt.subplots(figsize=(7.2,3.0),constrained_layout=True)
a.axhline(0,color="k",lw=0.6)
for stg,off,col in [("layer2",-0.16,SUP),("layer3",0.16,L3)]:
    for i,r in enumerate(ROIS):
        d=D[(D.roi==r)&(D.stage==stg)]
        for _,q in d.iterrows():
            sig=(q.ci_lo>0) or (q.ci_hi<0)
            a.scatter(i+off,q["diff"],marker=MK[q["sub"]],s=20,facecolors=col if sig else "none",edgecolors=col,lw=0.9,zorder=3)
        a.hlines(d["diff"].mean(),i+off-.12,i+off+.12,color="k",lw=1.4,zorder=4)
a.set_xticks(range(len(ROIS))); a.set_xticklabels(ROIS); a.set_ylabel("Δρ  (Supervised − SimCLR)")
h=[Line2D([],[],marker="o",ls="",color=SUP,label="layer2"),Line2D([],[],marker="o",ls="",color=L3,label="layer3"),
   Line2D([],[],marker="o",ls="",mfc="gray",mec="gray",label="95% CI excludes 0"),
   Line2D([],[],marker="o",ls="",mfc="none",mec="gray",label="95% CI includes 0"),
   Line2D([],[],color="k",lw=1.4,label="Mean over subjects")]+\
  [Line2D([],[],marker=MK[s],ls="",mfc="none",mec="gray",label=s) for s in SUBS]
fig.legend(handles=h,loc="upper center",bbox_to_anchor=(0.5,0.0),ncol=5,frameon=False,fontsize=6.5)
save(fig,"pub2_figE_delta_rho")
print("done")
