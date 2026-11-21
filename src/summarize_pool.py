import re, glob, pandas as pd, numpy as np
subs = ["sub-01","sub-02","sub-03","sub-06"]
dfs=[]
for s in subs:
    try:
        d=pd.read_csv(f"results/{s}_rsa_pooling.csv"); d["sub"]=s; dfs.append(d)
    except FileNotFoundError: print("missing",s)
df=pd.concat(dfs)
def parse(r):
    m=re.match(r"(stem|layer\d)(?:_b(\d))?(?:_g(\d))?$",r)
    if r=="avgpool": return ("avgpool",None,None)
    return (m.group(1),m.group(2),m.group(3))
df[["stage","blk","g"]]=df.readout.apply(lambda r: pd.Series(parse(r)))
order={"stem":0,"layer1":1,"layer2":2,"layer3":3,"layer4":4,"avgpool":5}
df["so"]=df.stage.map(order)
df=df[~((df.sub=="sub-02")&(df.roi=="MPA"))]   # 5 voxels, unusable
df.to_csv("results/pooling_all_subjects_long.csv",index=False)

# 1. peak readout per sub x roi x model (any readout) and its stage
idx=df.groupby(["sub","roi","model"]).rho.idxmax()
pk=df.loc[idx,["sub","roi","model","readout","stage","rho"]]
pk.to_csv("results/summary_peak_readout.csv",index=False)
print("\nPEAK STAGE (count of subjects per roi x model):")
print(pk.groupby(["roi","model"]).stage.agg(lambda x:", ".join(sorted(x))).unstack())

# 2. pooling robustness: best-over-stage rho at each grid (stage-level only, no blocks)
st=df[df.blk.isna()&df.g.notna()].copy()
rob=st.groupby(["sub","roi","model","g"]).apply(lambda x:x.loc[x.rho.idxmax(),"stage"]).rename("peak_stage").reset_index()
print("\nPEAK STAGE BY GRID (sub-01):"); print(rob[rob["sub"]=="sub-01"].pivot_table(index=["roi","model"],columns="g",values="peak_stage",aggfunc="first"))

# 3. supervised - simclr by grid at layer2/layer3, sign consistency across subjects
w=st.pivot_table(index=["sub","roi","stage","g"],columns="model",values="rho").reset_index()
w["d_sup_simclr"]=w.supervised-w.simclr
w["d_sup_random"]=w.supervised-w.random
w.to_csv("results/summary_diff_by_grid.csv",index=False)
cons=w.groupby(["roi","stage","g"]).d_sup_simclr.agg(n_pos=lambda x:(x>0).sum(),n="count",mean="mean").reset_index()
print("\nSUP>SIMCLR sign consistency (n_pos/n subjects), layer2/3:")
print(cons[cons.stage.isin(["layer2","layer3"])].pivot_table(index=["roi","stage"],columns="g",values="n_pos"))

# 4. block-level curves (g1) mean across subjects
bl=df[df.blk.notna()&(df.g=="1")].copy()
bl["block"]=bl.stage+"_b"+bl.blk
bc=bl.groupby(["roi","model","block"]).rho.mean().unstack("block")
bc=bc[sorted(bc.columns,key=lambda c:(int(c[5]),int(c[-1])))]
bc.to_csv("results/summary_block_curves_g1.csv")
print("\nBlock curve table saved: results/summary_block_curves_g1.csv")
