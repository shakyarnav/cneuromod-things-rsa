import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.linewidth":0.6,
    "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})
SUP,SIM,RND="#0072B2","#E69F00","#8C8C8C"
CM={"supervised":SUP,"simclr":SIM,"random":RND}
NM={"supervised":"Supervised","simclr":"SimCLR","random":"Random init"}
ROIS=["V1","V2","V3","OFA","FFA","EBA","OPA","PPA","MPA"]
STG=["stem","layer1","layer2","layer3","layer4"]; COLS=STG+["avgpool"]
LAB=["stem","layer1","layer2","layer3","layer4","avgpool"]
SUBS=["sub-01","sub-02","sub-03","sub-06"]; MK=dict(zip(SUBS,["o","s","^","D"]))
df=pd.read_csv("results/pooling_all_subjects_long.csv")
st=df[df.blk.isna()&df.g.isin([1,2,4])&df.stage.isin(STG)]
per=st.groupby(["sub","roi","model","stage"]).rho.max().reset_index()
ap=df[df.stage=="avgpool"][["sub","roi","model","stage","rho"]]
per=pd.concat([per,ap],ignore_index=True)
ceil=df.groupby("roi").ceiling.mean()
def save(fig,n): fig.savefig(f"figures/{n}.pdf"); fig.savefig(f"figures/{n}.png",dpi=300); plt.close(fig)

# ---- Fig 1: hierarchy heatmaps
fig,axs=plt.subplots(1,3,figsize=(7.2,3.1),sharey=True)
for a,m in zip(axs,["supervised","simclr","random"]):
    M=per[per.model==m].groupby(["roi","stage"]).rho.mean().unstack().reindex(index=ROIS,columns=COLS)
    im=a.imshow(M.values,cmap="viridis",vmin=0,vmax=0.40,aspect="auto")
    for i in range(len(ROIS)):
        for j in range(len(COLS)):
            v=M.values[i,j]
            if np.isfinite(v): a.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=6,color="w" if v<0.22 else "k")
        pk=int(np.nanargmax(M.values[i,:5])); a.add_patch(Rectangle((pk-.5,i-.5),1,1,fill=False,ec="red",lw=1.2))
    a.set_xticks(range(6)); a.set_xticklabels(LAB,rotation=45,ha="right"); a.set_title(NM[m],color=CM[m],fontweight="bold")
    a.spines[:].set_visible(False)
axs[0].set_yticks(range(len(ROIS))); axs[0].set_yticklabels(ROIS)
fig.colorbar(im,ax=axs,fraction=0.025,pad=0.02,label="RSA ρ (Spearman)")
fig.text(0.01,0.97,"A",fontweight="bold",fontsize=11)
save(fig,"pub_fig1_hierarchy")

# ---- Fig 2: per-subject profiles
R2=["V1","V3","OFA","FFA","EBA","PPA"]
fig,axs=plt.subplots(2,3,figsize=(7.2,4.4),sharex=True)
for a,r in zip(axs.flat,R2):
    for m in ["random","simclr","supervised"]:
        p=per[(per.roi==r)&(per.model==m)].pivot_table(index="stage",columns="sub",values="rho").reindex(COLS)
        if m!="random":
            for s in p.columns: a.plot(range(6),p[s],color=CM[m],lw=0.6,alpha=0.35)
        a.plot(range(6),p.mean(1),color=CM[m],lw=1.8 if m!="random" else 1.2,ls="-" if m!="random" else "--",label=NM[m])
    a.axhline(ceil[r],color="k",ls=":",lw=0.8)
    a.set_title(f"{r} (n={p.shape[1]})",fontsize=8); a.set_xticks(range(6)); a.set_xticklabels(LAB,rotation=45,ha="right")
    a.set_ylim(0,max(0.6,ceil[r]+0.03)) if False else None
for a in axs[:,0]: a.set_ylabel("RSA ρ")
axs[0,0].legend(frameon=False,fontsize=6.5)
fig.tight_layout(); save(fig,"pub_fig2_profiles_by_subject")

# ---- Fig 3: architecture vs learning, labels vs self-supervision
pk=st.groupby(["sub","roi","model"]).rho.max().reset_index()
D=[]
for s in SUBS:
    d=pd.read_csv(f"results/{s}_rsa_pooling_diff.csv"); d["sub"]=s; D.append(d)
D=pd.concat(D); D=D[~((D["sub"]=="sub-02")&(D.roi=="MPA"))]
D=D[(D.contrast=="supervised_minus_simclr")&(D.grid==2)&(D.block==0)&D.stage.isin(["layer2","layer3"])]
fig,(a1,a2)=plt.subplots(2,1,figsize=(7.2,5.2))
x=np.arange(len(ROIS))
for j,m in enumerate(["random","simclr","supervised"]):
    off=(j-1)*0.26
    for i,r in enumerate(ROIS):
        d=pk[(pk.roi==r)&(pk.model==m)]
        for _,q in d.iterrows(): a1.scatter(i+off,q.rho,marker=MK[q["sub"]],s=14,color=CM[m],alpha=0.8,lw=0)
        a1.hlines(d.rho.mean(),i+off-.1,i+off+.1,color="k",lw=1.2)
    a1.scatter([],[],color=CM[m],label=NM[m],s=20)
a1.set_xticks(x); a1.set_xticklabels(ROIS); a1.set_ylabel("Peak RSA ρ (best stage)"); a1.legend(frameon=False,ncol=3,loc="upper right")
a1.text(-0.07,1.02,"A",transform=a1.transAxes,fontweight="bold",fontsize=11)
a2.axhline(0,color="k",lw=0.6)
for k,(stg,off) in enumerate([("layer2",-0.15),("layer3",0.15)]):
    for i,r in enumerate(ROIS):
        d=D[(D.roi==r)&(D.stage==stg)]
        for _,q in d.iterrows():
            sig=(q.ci_lo>0) or (q.ci_hi<0)
            a2.scatter(i+off,q["diff"],marker=MK[q["sub"]],s=18,facecolors=(SUP if k==0 else "#56B4E9") if sig else "none",
                       edgecolors=SUP if k==0 else "#56B4E9",lw=0.8)
        a2.hlines(d["diff"].mean(),i+off-.1,i+off+.1,color="k",lw=1.2)
    a2.scatter([],[],color=SUP if k==0 else "#56B4E9",s=20,label=stg)
a2.set_xticks(x); a2.set_xticklabels(ROIS); a2.set_ylabel("Δρ = Supervised − SimCLR")
a2.legend(frameon=False,ncol=2,loc="lower left",title="filled: subject's 95% category-bootstrap CI excludes 0",title_fontsize=6.5)
a2.text(-0.07,1.02,"B",transform=a2.transAxes,fontweight="bold",fontsize=11)
for m_,s_ in MK.items(): a2.scatter([],[],marker=s_,color="gray",s=14,label=m_)
a2.legend(frameon=False,ncol=3,loc="lower right",fontsize=6.5)
fig.tight_layout(); save(fig,"pub_fig3_architecture_vs_learning")
print("done")
