"""Build a self-contained Colab notebook and execute it on the local GPU."""
import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager


root=Path(__file__).resolve().parent
os.environ['JUPYTER_RUNTIME_DIR']=str(root/'tmp/jupyter')
os.environ['IPYTHONDIR']=str(root/'tmp/ipython')
os.environ['MPLCONFIGDIR']=str(root/'.cache/matplotlib')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(parents=True,exist_ok=True)
nb=nbf.v4.new_notebook()
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'},'colab':{'name':'HW4_Part2_Colab.ipynb'},'execution_note':'Executed locally on the RTX 4070; not executed in Google Colab. Rerun in Colab with a GPU runtime.'}
nb.cells=[
nbf.v4.new_markdown_cell('''# HW4 Part 2: Medical image transfer learning

**Model:** ImageNet-pretrained ResNet-18. **Dataset:** DermaMNIST (MedMNIST / HAM10000), 7 classes.

This notebook includes all training code and saves predictions, checkpoint, learning curves, ROC curves, confusion matrices, t-SNE and a machine-readable result file. Saved outputs were produced on a **local NVIDIA RTX 4070**, not in Google Colab. To rerun in Colab, choose **Runtime > Change runtime type > GPU** and run all cells. GPU/CPU and package differences can change results.

We retain the official 7,007 / 1,003 / 2,005 train / validation / test split. Only training images are augmented. Train-only class weights handle imbalance. Epoch 1 trains the new head; epochs 2-6 fine-tune layer4 and the head with frozen BatchNorm statistics. The best validation-accuracy checkpoint is selected before the test set is evaluated.

Sources: [MedMNIST](https://medmnist.com/) · [official code](https://github.com/MedMNIST/MedMNIST) · [torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
'''),
nbf.v4.new_code_cell('''import sys, subprocess
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'numpy==2.2.6', 'scipy', 'matplotlib', 'scikit-learn', 'medmnist==3.0.2'])
print('Python:', sys.version)
'''),
nbf.v4.new_code_cell("from pathlib import Path\n__file__ = str(Path.cwd() / 'finetune.py')\n"+ (root/'finetune.py').read_text(encoding='utf-8').replace("if __name__=='__main__': main()", "import sys\nsys.argv = ['finetune.py', '--epochs', '6']\nmain()")),
nbf.v4.new_code_cell('''import json
from IPython.display import display, Image, Markdown
result = json.loads(Path('results/finetune/result.json').read_text())
val = result['validation']; test = result['test']
report = json.loads(Path('results/finetune/val_classification_report.json').read_text())
majority_name = list(result['dataset_info']['label'].values())[int(__import__('numpy').argmax(result['train_class_counts']))]
majority_val = report[majority_name]['support'] / result['split_sizes']['val']
display(Markdown(f"""## Results and interpretation
- Best epoch: **{result['best_epoch']}** (validation accuracy selection).
- Validation accuracy: **{val['accuracy']:.2%}**; balanced accuracy: **{val['balanced_accuracy']:.2%}**; macro one-vs-rest AUC: **{val['macro_ovr_auc']:.3f}**.
- Held-out test accuracy: **{test['accuracy']:.2%}**; balanced accuracy: **{test['balanced_accuracy']:.2%}**; macro AUC: **{test['macro_ovr_auc']:.3f}**.
- Always predicting the training-majority class achieves **{majority_val:.2%}** validation accuracy but **14.29%** balanced accuracy. The model's overall accuracy is close to that baseline; its higher balanced accuracy shows better minority-class coverage. Raw accuracy alone is insufficient.
- Native images are 28 x 28; resizing to 224 x 224 does not recover clinical detail. Official splits are retained, but raw patient identity leakage was not independently audited. One dataset and seed do not support a clinical claim. t-SNE is descriptive, not an accuracy score.
"""))
for name in ['learning_curves.png', 'val_confusion.png', 'val_roc.png', 'val_tsne.png', 'test_confusion.png', 'test_roc.png']:
    display(Image(filename='results/finetune/'+name))
'''),
nbf.v4.new_code_cell('''# Colab: download the result files after running. Do not upload the dataset to GitHub.
import shutil
shutil.make_archive('HW4_Part2_results', 'zip', 'results/finetune')
if 'google.colab' in sys.modules:
    from google.colab import files
    files.download('HW4_Part2_results.zip')
print('Saved HW4_Part2_results.zip')
''')]
km=KernelManager(kernel_name='python3',connection_file=str(root/'tmp/jupyter/kernel_hw4.json'))
km.kernel_spec.argv=[str(root/'.venv/Scripts/python.exe'),'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(nb,km=km,timeout=600,resources={'metadata':{'path':str(root)}})
try:
    client.execute()
finally:
    nbf.write(nb,root/'HW4_Part2_Colab.ipynb')
if any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs):
    raise RuntimeError('Notebook contains an execution error')
print('Executed and saved HW4_Part2_Colab.ipynb')
