LaTeX report using the supplied HW2 report layout.
Author: Yuntian Wu, yuntianw@andrew.cmu.edu.

Overleaf: upload this ZIP as a new project and set main.tex as the main document.
Compile with pdfLaTeX (twice for the table of contents), XeLaTeX, or Tectonic.
The figures directory and all input .tex files must remain beside main.tex.
manifest.json records the hashes of the matching source files, figures, and PDF.

The report includes actual heart-rate analysis, medical-image training, eight
physical 1 bit/s optical trials, and 2/3 bit/s trials at 120 cm / delta alpha 0.1.
Both faster rates remained visibly flickering, as reported by the participant.
Phone videos were decoded offline. A separate live phone screen recording
documents CMU with CRC PASS; the report includes extracted progress/success
frames. The recorded demo uses configured 20 cm, alpha difference 0.1 and 1 bit/s.
No raw live intensity CSV or demo transmitter log is archived. Both interfaces
are English. The unmodified video is provided beside the report in the demo pack.
Public GitHub repository:
https://github.com/GitaTReNt/17-720-HW4-Vital-Visions

In the original homework directory, python build_latex_report.py stages the
figures and compiles the matching PDF and Overleaf ZIP. The local portable
Tectonic executable is kept outside this source package, in .tools/tectonic.
