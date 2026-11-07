"""Refresh notebook charts from saved predictions without another training run."""
import os
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
root=Path(__file__).resolve().parent
os.environ['JUPYTER_RUNTIME_DIR']=str(root/'tmp/jupyter')
os.environ['IPYTHONDIR']=str(root/'tmp/ipython')
os.environ['MPLCONFIGDIR']=str(root/'.cache/matplotlib')
path=root/'HW4_Part2_Colab.ipynb';nb=nbf.read(path,as_version=4)
render=nbf.v4.new_code_cell("# Improve figure label readability from saved predictions; no retraining.\nimport sys\nfrom pathlib import Path\n__file__ = str(Path.cwd()/'refresh_plots.py')\n"+(root/'refresh_plots.py').read_text(encoding='utf-8'))
tail=nbf.v4.new_notebook(cells=[render,nb.cells[3],nb.cells[4]])
km=KernelManager(kernel_name='python3',connection_file=str(root/'tmp/jupyter/kernel_render.json'))
km.kernel_spec.argv=[str(root/'.venv/Scripts/python.exe'),'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(tail,km=km,timeout=300,resources={'metadata':{'path':str(root)}}).execute()
nb.cells=nb.cells[:3]+tail.cells
for index,cell in enumerate([c for c in nb.cells if c.cell_type=='code'],1):cell.execution_count=index
nbf.validate(nb);nbf.write(nb,path)
print('Updated readable notebook figures from saved predictions.')
