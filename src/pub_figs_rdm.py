import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
import os
ANN=os.environ.get("ANN","/scratch/arnav.shakya/cneuromod-things/THINGS/glmsingle/task-things_imgAnnotations.json")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.linewidth":0.6,"pdf.fonttype":42,"ps.fonttype":42})
SUB="sub-01"; FM="features/multi"
z=np.load(f"brain/{SUB}_category_rdms.npz",allow_pickle=True); cats=[str(c) for c in z["categories"]]
ann=json.load(open(ANN)); names=json.load(open(f"{FM}/supervised/images.json"))
first=ann[names[0]]; print("annotation keys:",list(first.keys()))
akey=[k for k in first if "anim" in k.lower() or "living" in k.lower()]
if not akey: raise SystemExit("No animacy-like key found; paste the keys printed above.")
akey=akey[0]; print("using animacy key:",akey,"| example:",first[akey])
catof=np.array([ann[n]["image_category"] for n in names])
def isanim(c):
    v=ann[names[list(catof).index(c)]][akey]
    return bool(v) if not isinstance(v,str) else v.lower() in ("1","true","animate","living","yes")
anim=np.array([isanim(c) for c in cats])
order=np.lexsort((np.array(cats),~anim))          # animate first, then alphabetical
n_anim=int(anim.sum()); print("animate categories:",n_anim,"of",len(cats))
idx=[np.where(catof==c)[0] for c in cats]
def model_rdm(model,readout):
    X=np.load(f"{FM}/{model}/{readout}.npy",mmap_mode="r"); X=np.asarray(X).reshape(len(names),-1).astype(np.float32)
    P=np.stack([X[i].mean(0) for i in idx]); P-=P.mean(1,keepdims=True); P/=np.linalg.norm(P,axis=1,keepdims=True)+1e-12
    return 1-P@P.T
iu=np.triu_indices(len(cats),1)
rk=lambda M: rankdata(M[iu])
L=pd.read_csv("results/pooling_all_subjects_long.csv")
S=L[(L["sub"]==SUB)&L.blk.isna()&L.g.isin([1,2,4])&L.stage.isin(["layer1","layer2","layer3","layer4"])&(L.model=="supervised")]
best={r:S[S.roi==r].sort_values("rho").iloc[-1] for r in ["V1","FFA","PPA"]}
def show(a,M,title):
    R=np.zeros_like(M); R[iu]=rankdata(M[iu])/len(iu[0]); R=R+R.T; R=R[np.ix_(order,order)]
    a.imshow(R,cmap="viridis",vmin=0,vmax=1,interpolation="nearest"); a.set_xticks([]); a.set_yticks([]); a.set_title(title,fontsize=7.5)
    for p in (n_anim-.5,): a.axhline(p,color="w",lw=.5); a.axvline(p,color="w",lw=.5)
# ---------- Fig F
fig,axs=plt.subplots(2,3,figsize=(6.6,4.6),constrained_layout=True)
for j,r in enumerate(["V1","FFA","PPA"]):
    show(axs[0,j],z[f"rdm_{r}"],f"{SUB} {r} (brain)")
    b=best[r]; show(axs[1,j],model_rdm("supervised",b.readout),f"Supervised {b.readout}\n(ρ = {b.rho:.2f} with {r})")
im=axs[0,0].images[0]; cb=fig.colorbar(im,ax=axs,fraction=0.03,pad=0.02); cb.set_label("Correlation distance (rank, 0–1)"); cb.outline.set_visible(False)
axs[1,0].set_xlabel(f"720 categories: animate (first {n_anim}) | inanimate",fontsize=7,loc="left")
fig.savefig("figures/pub2_figF_rdms.pdf",bbox_inches="tight"); fig.savefig("figures/pub2_figF_rdms.png",dpi=300,bbox_inches="tight"); plt.close(fig)
# ---------- Fig G: second-order RSA
items={}; cols={}
for r in ["V1","V2","V3","OFA","FFA","EBA","OPA","PPA","MPA"]: items[r]=rk(z[f"rdm_{r}"]); cols[r]="0.15"
for m,tag,c in [("supervised","Sup","#0072B2"),("simclr","SimCLR","#E69F00")]:
    for l in ["layer1","layer2","layer3","layer4"]:
        items[f"{tag} {l}"]=rk(model_rdm(m,f"{l}_g2")); cols[f"{tag} {l}"]=c
    items[f"{tag} avgpool"]=rk(model_rdm(m,"avgpool")); cols[f"{tag} avgpool"]=c
keys=list(items); A=np.stack([items[k] for k in keys]); C=np.corrcoef(A); Dm=1-C; np.fill_diagonal(Dm,0)
emb=MDS(2,dissimilarity="precomputed",random_state=0,n_init=8).fit(Dm); xy=emb.embedding_
fig,(a,b)=plt.subplots(1,2,figsize=(7.2,3.6),constrained_layout=True,gridspec_kw={"width_ratios":[1,1.1]})
for k,(x,y) in zip(keys,xy):
    a.scatter(x,y,s=28,color=cols[k],marker="s" if k[:1]!="S" and k[:3]!="Sim" else "o",zorder=3); a.annotate(k,(x,y),fontsize=5.5,xytext=(3,3),textcoords="offset points")
a.set_xlabel("MDS dim 1"); a.set_ylabel("MDS dim 2"); a.set_title(f"MDS (stress = {emb.stress_:.1f})",fontsize=8)
a.text(-0.12,1.04,"A",transform=a.transAxes,fontweight="bold",fontsize=11)
Z=linkage(squareform(Dm,checks=False),"average"); d=dendrogram(Z,labels=keys,orientation="right",ax=b,leaf_font_size=6,color_threshold=0,above_threshold_color="0.3")
for t in b.get_ymajorticklabels(): t.set_color(cols[t.get_text()])
b.set_xlabel("1 − Spearman ρ between RDMs"); b.set_title("Average-linkage clustering",fontsize=8)
b.text(-0.05,1.04,"B",transform=b.transAxes,fontweight="bold",fontsize=11)
for s in ("top","right"): a.spines[s].set_visible(False); b.spines[s].set_visible(False)
fig.savefig("figures/pub2_figG_second_order.pdf",bbox_inches="tight"); fig.savefig("figures/pub2_figG_second_order.png",dpi=300,bbox_inches="tight")
np.save("results/second_order_dist_sub01.npy",Dm); json.dump(keys,open("results/second_order_keys_sub01.json","w"))
print("done")
