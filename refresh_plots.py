"""Render compact, readable class labels from saved predictions; no retraining."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,roc_curve,roc_auc_score
from sklearn.manifold import TSNE

out=Path('results/finetune');names=['AKIEC','BCC','BKL','DF','MEL','NV','VASC']
for split in ['val','test']:
    r=np.load(out/f'{split}_predictions.npz');y=r['labels'];prob=r['probabilities']
    fig,ax=plt.subplots(figsize=(7,6))
    ConfusionMatrixDisplay(confusion_matrix(y,prob.argmax(1),labels=range(7)),display_labels=names).plot(ax=ax,xticks_rotation=0,colorbar=False,values_format='d')
    ax.set_title(f'DermaMNIST {split}: ImageNet ResNet-18');fig.tight_layout();fig.savefig(out/f'{split}_confusion.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for k,name in enumerate(names):
        fpr,tpr,_=roc_curve(y==k,prob[:,k]);auc=roc_auc_score(y==k,prob[:,k]);ax.plot(fpr,tpr,label=f'{name}: {auc:.3f}')
    ax.plot([0,1],[0,1],'k--',alpha=.4);ax.set(xlabel='1 - specificity',ylabel='Sensitivity',title=f'{split}: one-vs-rest ROC');ax.legend(fontsize=9);fig.tight_layout();fig.savefig(out/f'{split}_roc.png',dpi=170);plt.close(fig)
    if split=='val':
        emb=TSNE(n_components=2,random_state=42,perplexity=30,init='pca',learning_rate='auto').fit_transform(r['features'])
        fig,ax=plt.subplots(figsize=(7,5))
        for k,name in enumerate(names):ax.scatter(*emb[y==k].T,s=9,alpha=.65,label=name)
        ax.set(title='Validation: t-SNE of penultimate features',xlabel='t-SNE 1',ylabel='t-SNE 2');ax.legend(fontsize=9);fig.tight_layout();fig.savefig(out/'val_tsne.png',dpi=170);plt.close(fig)
print('Refreshed plots from saved predictions; model and metrics unchanged.')
