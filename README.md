# motor-imagery-analysis
Train a left/right motor imagery EEG classifier and generate predictions for unseen test trials.
## Report
`final_report.pdf` — final write-up and results.
## Project structure
- `main.py` — runs the full pipeline (visualization → feature extraction → classification → test predictions).
- `visualization.py` — data loading, filtering valid LEFT/RIGHT trials, and plotting examples.
- `power_spectra.py` — Welch PSD and spectrogram analysis utilities.
- `features.py` — feature extraction (e.g., band-power) + PCA helpers.
- `classification.py` — LDA-based evaluation (CV) and test prediction generation.
- ## How to run
Dataset files (`motor_imagery_train_data.npy`, `motor_imagery_test_data.npy`) are not included (course materials).
Place them locally in the project root and run:
python main.py

