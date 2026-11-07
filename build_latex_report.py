"""Stage measured figures and compile the report in the user-supplied HW2 layout."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'latex_report'
FIGURES = SOURCE / 'figures'
BUILD = SOURCE / 'build'
OUT = ROOT / 'output/pdf'

FIGURE_SOURCES = {
    'heart_fft.png': 'results/heart_rate/IMG_2324/fft.png',
    'heart_peaks.png': 'results/heart_rate/IMG_2324/heartbeat_peaks.png',
    'learning_curves.png': 'results/finetune/learning_curves.png',
    'val_confusion.png': 'results/finetune/val_confusion.png',
    'val_roc.png': 'results/finetune/val_roc.png',
    'val_tsne.png': 'results/finetune/val_tsne.png',
    'baseline_matrix.png': 'results/communication/baseline_matrix.png',
    'd80_transmissions.png': 'results/communication/d80_continuous_review/both_transmissions.png',
    'fast_transmissions.png': 'results/communication/d120_fast_continuous_review/both_transmissions.png',
    'live_progress.png': 'results/communication/live_demo/frame_40s.png',
    'live_success.png': 'results/communication/live_demo/frame_70s.png',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    for directory in [FIGURES, BUILD, OUT]:
        directory.mkdir(parents=True, exist_ok=True)
    hr = json.loads((ROOT/'results/heart_rate/IMG_2324/result.json').read_text())
    model = json.loads((ROOT/'results/finetune/result.json').read_text())
    fast = json.loads((ROOT/'results/communication/d120_fast_continuous_review/summary.json').read_text())
    live = json.loads((ROOT/'results/communication/real_time_reception.json').read_text())
    assert round(hr['fft_bpm'], 2) == 62.16
    assert round(model['validation']['accuracy']*100, 2) == 66.60
    assert [round(r['nominal_raw_bps']) for r in fast['transmissions']] == [2, 3]
    assert all(r['full_packet_within_segment'] and r['correct_bits'] == 53 and r['crc_ok'] for r in fast['transmissions'])
    for name, relative in FIGURE_SOURCES.items():
        shutil.copy2(ROOT/relative, FIGURES/name)
    env = os.environ.copy()
    env['TECTONIC_CACHE_DIR'] = str(ROOT/'.cache/tectonic')
    env['XDG_CACHE_HOME'] = str(ROOT/'.cache')
    if (SOURCE/'fonts.conf').exists():
        env['FONTCONFIG_FILE'] = str(SOURCE/'fonts.conf')
    local_engine = ROOT/'.tools/tectonic/tectonic.exe'
    engine = str(local_engine) if local_engine.exists() else shutil.which('tectonic')
    if engine:
        commands = [[engine, '--untrusted', '--keep-logs', '--keep-intermediates', '--outdir', str(BUILD), 'main.tex']]
    else:
        engine = shutil.which('pdflatex') or shutil.which('xelatex')
        if not engine:
            raise RuntimeError('Install Tectonic or a TeX distribution, or compile latex_report/main.tex in Overleaf.')
        commands = [[engine, '-no-shell-escape', '-halt-on-error', '-interaction=nonstopmode', '-output-directory', str(BUILD), 'main.tex']]*2
    output=[]
    for cmd in commands:
        run = subprocess.run(cmd, cwd=SOURCE, env=env, text=True, encoding='utf-8', errors='replace', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output.append(run.stdout)
        (BUILD/'compile_output.txt').write_text('\n'.join(output), encoding='utf-8')
        print(run.stdout)
        run.check_returncode()
    log = (BUILD/'main.log').read_text(encoding='utf-8', errors='replace')
    for forbidden in ['Overfull', 'Missing character:', 'undefined references', 'LaTeX Error:']:
        if forbidden in log:
            raise RuntimeError(f'LaTeX log requires review: {forbidden}')
    pdf = OUT/'HW4_Report_Yuntian_Wu.pdf'
    shutil.copy2(BUILD/'main.pdf', pdf)
    shutil.copy2(pdf, OUT/'HW4_report_DRAFT.pdf')
    files = [SOURCE/'main.tex', SOURCE/'fast_results.tex', SOURCE/'visibility.tex', SOURCE/'live_demo.tex', SOURCE/'README.txt'] + sorted(FIGURES.glob('*.png'))
    provenance=json.loads((ROOT/'sources/report_template_provenance.json').read_text())
    repository=json.loads((ROOT/'repository.json').read_text()) if (ROOT/'repository.json').exists() else {}
    manifest = {
        'template': 'hw2_report.tex from user-supplied Report___HW2___Yuntian_Wu.zip',
        'template_archive_sha256': provenance['sha256'],
        'engine': Path(engine).name+'; shell escape disabled',
        'pdf_sha256': sha(pdf),
        'phone_real_time_reception_demonstrated': live['status'] in ['user_confirmed_live_reception_success','recorded_live_reception_success'],
        'phone_real_time_reception_evidence': live['evidence_type'],
        'github_url': repository.get('url'),
        'files': [{'path': p.relative_to(SOURCE).as_posix(), 'sha256': sha(p)} for p in files],
    }
    (SOURCE/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    package = ROOT/'HW4_LaTeX_Overleaf.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in files+[SOURCE/'manifest.json']:
            archive.write(file, file.relative_to(SOURCE))
        archive.write(pdf, pdf.name)
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        assert archive.read(pdf.name) == pdf.read_bytes()
        for entry in manifest['files']:
            assert hashlib.sha256(archive.read(entry['path'])).hexdigest() == entry['sha256']
    print(json.dumps({'pdf': str(pdf), 'overleaf_zip': str(package), 'pdf_sha256': sha(pdf), 'zip_crc': 'passed'}, indent=2))

if __name__ == '__main__':
    main()
