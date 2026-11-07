# HW4: Vital Visions

**Yuntian Wu · CMU 17-720**

Code, executed notebook, measured results, and the LaTeX report for camera-based heart-rate estimation, medical-image fine-tuning, and alpha-channel screen-camera communication.

## Report and notebook

- [Report PDF](output/pdf/HW4_Report_Yuntian_Wu.pdf)
- [LaTeX source](latex_report/main.tex)
- [Executed, self-contained Colab-compatible notebook](HW4_Part2_Colab.ipynb)
- [Physical communication trial summary](results/communication/baseline_summary.json)
- [Recorded live receiver evidence](results/communication/real_time_reception.json)

The original demo video is provided with the report in the local submission package. Raw phone videos, the dataset archive, model weights, temporary tunnel logs, and machine runtimes are excluded from this repository. Extracted demo evidence frames are included in the report.

## Results and evidence

| Part | Experiment | Recorded result |
|---|---|---|
| 1 | Finger-over-camera video with flash | FFT: 62.16 bpm; peak intervals: 62.15 bpm |
| 2 | ImageNet ResNet-18 fine-tuned on DermaMNIST | Validation accuracy: 66.60%; test accuracy: 67.73% |
| 3 | 20/40/80/120 cm × alpha difference 0.1/0.5 at 1 bit/s | Eight 53-bit packets decoded as `CMU`, CRC passed |
| 3 | 120 cm, alpha difference 0.1 at 2 and 3 bit/s | Both packets decoded as `CMU`, CRC passed |
| 3 | Additional live phone receiver demo | Video shows `Received: CMU | CRC PASS` |

The ten offline packets contain 530 compared frame bits with no observed errors. The 20 cm / alpha 0.1 baseline lacks its original transmitter log: its BER comparison is explicitly conditional on the independently predeclared `CMU` protocol. One packet per condition does not establish zero long-run error probability. Display geometry varies across distance recordings. The live screen-recorded demo is separate from the offline BER comparison and does not add new BER measurements.

Both alpha differences at the baseline rate, and both faster tested rates, remained visibly flickering. Imperceptible communication and channel capacity were not demonstrated.

Fine-tuning ran locally on an RTX 4070, not in a Colab cloud session. The notebook retains actual execution outputs and can be uploaded to Colab. The validation majority-class baseline is 66.90% accuracy; the model's 66.60% accuracy does not exceed it, while balanced accuracy is higher (53.81% versus 14.29%). This is an educational experiment, not a clinically validated diagnostic system.

## Run the code

Python 3.11 was used. Install dependencies in a virtual environment:

```sh
python -m pip install -r requirements.txt
```

Analyze a locally supplied finger video:

```sh
python heart_rate.py path/to/finger_video.MOV --out results/my_heart_rate
```

Re-run the six-epoch medical-image experiment (downloads the official dataset and pretrained weights):

```sh
python finetune.py --epochs 6
```

Start the browser transmitter:

```sh
python -m http.server 8774 --bind 127.0.0.1 --directory web
```

Open `http://127.0.0.1:8774/transmitter.html`. The phone's live receiver requires an HTTPS deployment of the receiver HTML and JavaScript files. A computer's `127.0.0.1` URL does not address that computer from a phone.

Offline decoding accepts a video plus its original transmitter JSON and an image-interior ROI, or a saved receiver intensity CSV. See:

```sh
python communication.py --help
python decode_sequence.py --help
python decode_unlogged.py --help
```

## Checks

```sh
python verify_algorithms.py
node web/verify_codec.cjs
node web/verify_receiver_capture.cjs
node verify_recorded_codec.cjs
```

Software and synthetic checks are explicitly distinguished from physical experiments. `verify_recorded_codec.cjs` replays the recorded physical intensity traces through the browser decoder.

## Build the report

Upload `latex_report` to Overleaf and compile `main.tex` with pdfLaTeX, or install Tectonic / a TeX distribution and run:

```sh
python build_latex_report.py
```

For PDF rendering and artifact checks, also install `requirements-report.txt`, then run `render_latex_review.py` and `verify_latex_package.py`.

## Data and references

- [Assignment](https://canvas.cmu.edu/courses/55703/assignments/1008627)
- [MedMNIST](https://medmnist.com/) / [official code](https://github.com/MedMNIST/MedMNIST). DermaMNIST data license: CC BY-NC 4.0. Dataset files are not redistributed here.
- [Torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [HiLight paper linked by the assignment](https://cs.columbia.edu/~xia/publication/mobisys15-hilight/mobisys15-hilight.pdf)

The report follows the author's supplied HW2 LaTeX layout. No raw HW2 assignment archive is redistributed.
