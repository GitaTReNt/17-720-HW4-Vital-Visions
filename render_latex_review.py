"""Render the actual LaTeX PDF for visual inspection, and report page text."""
from pathlib import Path
import json
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
pdf = root/'latex_report/build/main.pdf'
out = root/'tmp/pdfs/latex_review'
out.mkdir(parents=True, exist_ok=True)
reader = PdfReader(pdf)
doc = pdfium.PdfDocument(pdf)
sheet = Image.new('RGB', (4*330, ((len(doc)+3)//4)*490), '#dfe3e8')
draw = ImageDraw.Draw(sheet)
stats=[]
for i, page in enumerate(doc):
    im=page.render(scale=1.5).to_pil().convert('RGB')
    im.save(out/f'page_{i+1:02}.png')
    im.thumbnail((310,450))
    x=(i%4)*330+10; y=(i//4)*490+25
    sheet.paste(im,(x,y));draw.text((x,y-18),f'Page {i+1}',fill='black')
    text=reader.pages[i].extract_text()
    stats.append({'page':i+1,'chars':len(text),'begin':text[:120].replace('\n',' '),'end':text[-90:].replace('\n',' ')})
sheet.save(out/'contact.png')
(out/'page_stats.json').write_text(json.dumps(stats,indent=2))
print(json.dumps(stats,indent=2))
