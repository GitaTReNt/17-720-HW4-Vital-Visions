"""ImageNet ResNet-18 transfer learning on the official DermaMNIST split."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('TORCH_HOME', str(Path(__file__).resolve().parent/'.cache/torch'))
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import medmnist
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.manifold import TSNE
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms


def evaluate(model, loader, device):
    model.eval()
    labels, probabilities, features = [], [], []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            f = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x))))))))).flatten(1)
            prob = model.fc(f).softmax(1)
            labels.append(y.numpy().reshape(-1)); probabilities.append(prob.cpu().numpy()); features.append(f.cpu().numpy())
    y, prob, feat = np.concatenate(labels), np.concatenate(probabilities), np.concatenate(features)
    return y, prob, feat


def metrics(y, prob):
    return {'accuracy':float(accuracy_score(y, prob.argmax(1))),
            'balanced_accuracy':float(balanced_accuracy_score(y, prob.argmax(1))),
            'macro_ovr_auc':float(roc_auc_score(y, prob, multi_class='ovr', average='macro'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/dermamnist'))
    parser.add_argument('--out',type=Path,default=Path('results/finetune'))
    parser.add_argument('--epochs',type=int,default=6)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    if args.epochs < 2:
        parser.error('Use at least 2 epochs so backbone fine-tuning actually occurs.')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.data.mkdir(parents=True,exist_ok=True); args.out.mkdir(parents=True,exist_ok=True)
    info=medmnist.INFO['dermamnist']; names=list(info['label'].values())
    short_names=['AKIEC','BCC','BKL','DF','MEL','NV','VASC']
    weights=models.ResNet18_Weights.IMAGENET1K_V1
    normalize=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    train_tf=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ToTensor(),normalize])
    eval_tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalize])
    sets={s:medmnist.DermaMNIST(split=s,root=str(args.data),download=True,transform=train_tf if s=='train' else eval_tf) for s in ['train','val','test']}
    # Keep official splits intact. No augmented copies enter validation or test.
    loaders={s:DataLoader(d,batch_size=args.batch_size,shuffle=s=='train',num_workers=0) for s,d in sets.items()}
    model=models.resnet18(weights=weights)
    for par in model.parameters(): par.requires_grad=False
    model.fc=nn.Linear(model.fc.in_features,len(names)); model.to(device)
    counts=np.bincount(sets['train'].labels.reshape(-1),minlength=len(names))
    class_weights=counts.sum()/(len(names)*counts)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(class_weights,dtype=torch.float32,device=device))
    optimizer=torch.optim.AdamW(model.fc.parameters(),lr=1e-3,weight_decay=1e-4)
    best_score=-1.; best_state=None; history=[]; start=time.time()
    print(json.dumps({'device':str(device),'counts':{s:len(d) for s,d in sets.items()},'class_counts':counts.tolist()}),flush=True)
    for epoch in range(args.epochs):
        if epoch==1:
            for par in model.layer4.parameters(): par.requires_grad=True
            optimizer=torch.optim.AdamW([{'params':model.layer4.parameters(),'lr':1e-4},{'params':model.fc.parameters(),'lr':3e-4}],weight_decay=1e-4)
        # Frozen BatchNorm statistics stay at the pretrained values.
        model.train()
        for mod in model.modules():
            if isinstance(mod,nn.BatchNorm2d): mod.eval()
        total_loss=0.; num=0; weighted_denominator=0.
        for x,y in loaders['train']:
            x=x.to(device); y=y.reshape(-1).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(x),y); loss.backward(); optimizer.step()
            denominator=float(criterion.weight[y].sum())
            total_loss+=float(loss.detach())*denominator; weighted_denominator+=denominator; num+=len(x)
        y,prob,_=evaluate(model,loaders['val'],device)
        row={'epoch':epoch+1,'stage':'head' if epoch==0 else 'layer4_and_head','train_loss':total_loss/weighted_denominator,**{'val_'+k:v for k,v in metrics(y,prob).items()}}
        history.append(row); print(json.dumps(row),flush=True)
        if row['val_accuracy']>best_score:
            best_score=row['val_accuracy']; best_epoch=epoch+1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        (args.out/'history.json').write_text(json.dumps(history,indent=2))
    model.load_state_dict(best_state)
    torch.save({'model_state_dict':best_state,'classes':names,'best_epoch':best_epoch,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},args.out/'best_model.pt')
    outputs={}
    for split in ['val','test']:
        y,prob,feat=evaluate(model,loaders[split],device); outputs[split]=metrics(y,prob)
        np.savez_compressed(args.out/f'{split}_predictions.npz',labels=y,probabilities=prob,features=feat)
        (args.out/f'{split}_classification_report.json').write_text(json.dumps(classification_report(y,prob.argmax(1),labels=list(range(len(names))),target_names=names,output_dict=True,zero_division=0),indent=2))
        fig,ax=plt.subplots(figsize=(7,6)); ConfusionMatrixDisplay(confusion_matrix(y,prob.argmax(1),labels=range(len(names))),display_labels=short_names).plot(ax=ax,xticks_rotation=0,colorbar=False,values_format='d')
        ax.set_title(f'DermaMNIST {split}: ImageNet ResNet-18'); fig.tight_layout(); fig.savefig(args.out/f'{split}_confusion.png',dpi=170); plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,5))
        for k,name in enumerate(names):
            fpr,tpr,_=roc_curve(y==k,prob[:,k]); auc=roc_auc_score(y==k,prob[:,k]); ax.plot(fpr,tpr,label=f'{short_names[k]}: {auc:.3f}')
        ax.plot([0,1],[0,1],'k--',alpha=.4); ax.set(xlabel='1 - specificity',ylabel='Sensitivity',title=f'{split}: one-vs-rest ROC'); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(args.out/f'{split}_roc.png',dpi=170); plt.close(fig)
        if split=='val':
            emb=TSNE(n_components=2,random_state=args.seed,perplexity=30,init='pca',learning_rate='auto').fit_transform(feat)
            fig,ax=plt.subplots(figsize=(7,5))
            for k,name in enumerate(names): ax.scatter(*emb[y==k].T,s=9,alpha=.65,label=short_names[k])
            ax.set(title='Validation: t-SNE of 512-dimensional penultimate features',xlabel='t-SNE 1',ylabel='t-SNE 2'); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(args.out/'val_tsne.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5)); epochs=[r['epoch'] for r in history]
    axes[0].plot(epochs,[r['train_loss'] for r in history],'o-'); axes[0].set(xlabel='Epoch',ylabel='Weighted training loss')
    axes[1].plot(epochs,[r['val_accuracy'] for r in history],'o-',label='Accuracy'); axes[1].plot(epochs,[r['val_balanced_accuracy'] for r in history],'o-',label='Balanced accuracy'); axes[1].set(xlabel='Epoch',ylabel='Validation metric',ylim=(0,1)); axes[1].legend()
    fig.tight_layout(); fig.savefig(args.out/'learning_curves.png',dpi=170); plt.close(fig)
    dataset_file=args.data/'dermamnist.npz'
    result={'status':'completed_real_training','dataset':'DermaMNIST (MedMNIST), native 28x28 RGB resized to 224x224',
            'dataset_info':info,'dataset_sha256':hashlib.sha256(dataset_file.read_bytes()).hexdigest(),
            'model':'ResNet-18','pretrained_weights':'IMAGENET1K_V1','split_sizes':{s:len(d) for s,d in sets.items()},
            'seed':args.seed,'epochs':args.epochs,'best_epoch':best_epoch,'selection':'highest official validation accuracy; test evaluated only after checkpoint selection',
            'train_class_counts':counts.tolist(),'class_weights':class_weights.tolist(),'validation':outputs['val'],'test':outputs['test'],
            'elapsed_seconds':time.time()-start,'device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
            'versions':{'torch':torch.__version__,'medmnist':medmnist.__version__,'numpy':np.__version__},
            'limitations':['Native 28x28 images lose clinical detail; resizing cannot restore it.','One fixed seed and one dataset; no clinical deployment or robustness claim.','Official split retained; patient grouping was not independently audited from raw source IDs.','t-SNE is descriptive and is not an accuracy measure.']}
    (args.out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
