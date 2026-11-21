import pandas as pd
old=pd.read_csv("results/sub-01_rsa_overall.csv")
new=pd.read_csv("results/sub-01_rsa_pooling.csv")
old=old[old.layer.isin(["layer1","layer2","layer3","layer4"])].copy()
old["readout"]=old.layer+"_g2"
m=old.merge(new[["roi","model","readout","rho"]],on=["roi","model","readout"],suffixes=("_old","_new"))
m["diff"]=(m.rho_old-m.rho_new).abs()
print(len(m),"pairs; max |diff| =",m["diff"].max())
print(m.sort_values("diff",ascending=False).head(5)[["roi","model","readout","rho_old","rho_new","diff"]])
print("PASS" if m["diff"].max()<1e-4 else "FAIL")
