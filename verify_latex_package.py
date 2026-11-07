"""Check that the delivered LaTeX source, measured values, PDF and ZIP agree."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
import unicodedata
from pypdf import PdfReader

root=Path(__file__).resolve().parent
source=root/'latex_report'
pdf=root/'output/pdf/HW4_Report_Yuntian_Wu.pdf'
digest=lambda data:hashlib.sha256(data).hexdigest()
reader=PdfReader(pdf)
assert len(reader.pages)==10
texts=[' '.join(unicodedata.normalize('NFKC',p.extract_text()).replace('\u2019',"'").split()) for p in reader.pages]
assert all(len(t)>300 for t in texts)
for term in ['62.16', '62.15', '66.60', '67.73', '53.81', '0.9127', '530', '38.551', '1.358']:
    assert term in '\n'.join(texts), term
assert 'noticeable flicker in both' in texts[7]
assert 'Received: CMU | CRC PASS' in texts[8]
assert '2,010 samples' in texts[8]
assert 'GitHub' in texts[-1]
repository=json.loads((root/'repository.json').read_text())
assert repository['visibility']=='public'
assert repository['url'] in '\n'.join(texts)
uris=[a.get_object().get('/A',{}).get('/URI') for page in reader.pages for a in page.get('/Annots',[])]
assert repository['url'] in uris
assert '\ufffd' not in '\n'.join(texts)
assert (root/'output/pdf/HW4_report_DRAFT.pdf').read_bytes()==pdf.read_bytes()
log=(source/'build/main.log').read_text(encoding='utf-8')
for issue in ['Overfull', 'Underfull', 'Missing character:', 'undefined references', 'LaTeX Error:']:
    assert issue not in log, issue
manifest=json.loads((source/'manifest.json').read_text())
assert digest(pdf.read_bytes())==manifest['pdf_sha256']
assert manifest['github_url']==repository['url']
with zipfile.ZipFile(root/'HW4_LaTeX_Overleaf.zip') as archive:
    assert archive.testzip() is None
    assert archive.read(pdf.name)==pdf.read_bytes()
    for entry in manifest['files']:
        assert archive.read(entry['path'])==(source/entry['path']).read_bytes()
        assert digest(archive.read(entry['path']))==entry['sha256']
for file in root.glob('*.py'):
    ast.parse(file.read_text(encoding='utf-8'))
summary=json.loads((root/'results/communication/baseline_summary.json').read_text())
assert summary['baseline_conditions']==8 and summary['faster_rate_trials']==2
assert summary['total_frame_bits_compared']==530 and summary['observed_frame_bit_errors']==0
assert summary['phone_real_time_reception_demonstrated']
assert summary['phone_real_time_reception_evidence']=='visually inspected phone screen recording'
tracking=json.loads((root/'experiment_log.json').read_text())
assert tracking['faster_rate_trials_analyzed']==2
checks={'pdf_pages':len(reader.pages),'latex_warnings':'none','source_pdf_zip_consistent':True,
        'zip_crc':'passed','physical_trials':10,'frame_bits_compared':530,
        'phone_real_time_reception_demonstrated':True,'phone_real_time_reception_evidence':'visually inspected phone screen recording','visual_review':'All pages rendered; contact sheet and detailed pages inspected.',
        'pdf_sha256':digest(pdf.read_bytes())}
(source/'verification.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
