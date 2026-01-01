# Feature Extraction

This folder contains scripts for extracting audio features from speech recordings used in the **Robust Speech Commands** project. Feature extraction is modular and configurable via **Hydra**, allowing you to select STFT, Mel-spectrogram, and MFCC features.

---

## Available Features

1. **STFT (Short-Time Fourier Transform)**  
   - Computes the magnitude or dB spectrogram of audio signals.  
   - Parallelized using Joblib for faster processing on multiple CPU cores.  
   - **Configurable parameters:**
     - `n_fft` – FFT size for each frame  
     - `win_length` – Window length for FFT  
     - `hop_length` – Step size between frames  
     - `window` – Window type (e.g., `hann`)  
     - `center` – Whether to pad signal to center frames  

2. **Mel-spectrogram**  
   - Converts waveforms into Mel-scale spectrograms.  
   - Optionally converts power spectrogram to dB scale.  
   - Parallelized for efficiency.  
   - **Configurable parameters:**
     - `n_fft` – FFT size  
     - `win_length` – Window length  
     - `hop_length` – Step size  
     - `n_mels` – Number of Mel bins  
     - `window` – Window type  
     - `center` – Centering of frames  
     - `fmin` – Minimum frequency for Mel filterbank  
     - `fmax` – Maximum frequency for Mel filterbank  

3. **MFCC (Mel-Frequency Cepstral Coefficients)**  
   - Extracts `n_mfcc` coefficients, optionally with delta and delta-delta derivatives.  
   - Parallelized for batch processing.  
   - Fast single-waveform extraction available for real-time inference.  
   - **Configurable parameters:**
     - `n_fft`, `win_length`, `hop_length`, `n_mels`, `n_mfcc` – FFT, window, hop size, number of Mel bins, number of MFCC coefficients  
     - `window`, `center` – Window type and frame centering  
     - `fmin`, `fmax` – Frequency range for filterbank  
     - `mfcc_include_deltas` – Include first and second derivatives  

4. **Settings**  
   - `stft_to_db` – Convert STFT magnitude to dB  
   - `mel_to_db` – Convert Mel power to dB  
   - `mfcc_include_deltas` – Include delta and delta-delta for MFCC  
   - `n_jobs` – Number of CPU cores for parallel extraction (-1 uses all cores)  

---

## How to Configure Feature Extraction

You can modify feature extraction behavior directly via Hydra parameters when running `main.py`.  

- To disable MFCC deltas:

```bash
python main.py features.settings.mfcc_include_deltas=False
```

- To change extraction parameters, e.g., FFT size:

```bash
python main.py features.mfcc.n_fft=512
```

- You can also control which features are extracted using `feature_to_extract`.
By default, only MFCC is extracted:

```yaml
feature_to_extract: ["mfcc"]
```

- You can request multiple features, e.g., `["stft", "mel", "mfcc"]`:

```bash
python main.py feature_to_extract=[stft,mel,mfcc]
```

These settings determine what `extract_feature` in `src/data/feature_extraction.py` computes.
It returns a dictionary mapping feature names to NumPy arrays, e.g., `{"mfcc": np.ndarray, "mel": np.ndarray}`.

> ⚠️ **Important**: In the current training pipeline (pipelines/train_pipeline.py), features are extracted like this:

```bash
spec_train = extract_feature(X_train, config)["mfcc"]
spec_val   = extract_feature(X_val, config)["mfcc"]
spec_test  = extract_feature(X_test, config)["mfcc"]
```

The code always expects MFCC to be present.
So, for now, removing MFCC from `feature_to_extract` may crash the pipeline.
This will be improved in a future update to support flexible feature selections.

---

## Future Improvements

- Add support for training with multiple feature types simultaneously (STFT, Mel, MFCC)
- Additional features like Chroma

---

## References

- [Librosa MFCC](https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html) – Mel-Frequency Cepstral Coefficients  
- [Librosa Mel-spectrogram](https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html) – Mel-scale spectrograms  
- [Librosa STFT](https://librosa.org/doc/latest/generated/librosa.stft.html) – Short-Time Fourier Transform  
- [Joblib Parallel Processing](https://joblib.readthedocs.io/en/latest/) – Parallel computation for faster feature extraction
