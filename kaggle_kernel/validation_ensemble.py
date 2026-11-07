"""Validation-only, probability-averaging ablation; no FER2013 test images used."""
import csv, json
from pathlib import Path
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED=20261107; LABELS=["angry","disgust","fear","happy","neutral","sad","surprise"]; INPUT=Path("/kaggle/input"); OUT=Path("/kaggle/working/affectiveecho_ensemble_validation"); OUT.mkdir(exist_ok=True)
def root():
    for p in INPUT.rglob("train"):
        if p.is_dir() and all((p/x).is_dir() for x in LABELS): return p.parent
    raise FileNotFoundError("FER2013 not found")
rows=[]
for label in LABELS:
    rows += [(p, LABELS.index(label)) for p in sorted((root()/"train"/label).glob("*.jpg"))]
ids=list(range(len(rows))); _, val_ids=train_test_split(ids,test_size=.1,random_state=SEED,stratify=[y for _,y in rows]); val=[rows[i] for i in val_ids]
tf=transforms.Compose([transforms.Resize((224,224)),transforms.Grayscale(3),transforms.ToTensor(),transforms.Normalize((.485,)*3,(.229,)*3)])
class D(Dataset):
    def __len__(self): return len(val)
    def __getitem__(self,i):
        p,y=val[i]; return tf(Image.open(p).convert("L")),y,str(p)
loader=DataLoader(D(),128,num_workers=4,pin_memory=True); device="cuda"
mobile=models.mobilenet_v3_large(weights=None); mobile.classifier[3]=nn.Linear(mobile.classifier[3].in_features,7); mobile.load_state_dict(torch.load("mobilenet_v3_baseline.pt",map_location=device)); mobile.to(device).eval()
efficient_path=next(p for p in INPUT.rglob("best_model.pt") if "fer2013" not in str(p.parent).lower())
efficient=models.efficientnet_b0(weights=None); efficient.classifier[1]=nn.Linear(efficient.classifier[1].in_features,7); efficient.load_state_dict(torch.load(efficient_path,map_location=device)); efficient.to(device).eval()
truth=[]; pred=[]; output=[]
with torch.no_grad():
    for x,y,paths in loader:
        scores=.5*torch.softmax(mobile(x.to(device)),1)+.5*torch.softmax(efficient(x.to(device)),1); p=scores.argmax(1).cpu().tolist(); y=y.tolist(); truth+=y; pred+=p
        output += [{"sample_id":path,"y_true":LABELS[a],"y_pred":LABELS[b]} for path,a,b in zip(paths,y,p)]
pr,re,f1,_=precision_recall_fscore_support(truth,pred,average="macro",zero_division=0)
metrics={"task":"validation-only probability ensemble; test set used: 0","n_validation":len(truth),"accuracy":accuracy_score(truth,pred),"macro_precision":pr,"macro_recall":re,"macro_f1":f1,"cohen_kappa":cohen_kappa_score(truth,pred)}
(OUT/"validation_metrics.json").write_text(json.dumps(metrics,indent=2));
with (OUT/"validation_predictions.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=output[0].keys());w.writeheader();w.writerows(output)
print(json.dumps(metrics,indent=2))
