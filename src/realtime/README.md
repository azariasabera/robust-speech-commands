# Real-time Keyword Spotting

This folder provides scripts for **real-time keyword spotting** using microphone input.
Audio is processed in short hops, aggregated over a sliding buffer, and passed through a trained model to detect spoken commands live.

---

## Usage

Run the realtime pipeline:

```bash
python main.py pipeline=[realtime]
```

This starts listening to the default microphone and prints detected commands to the console.

**Default behavior:**

* Sample rate: 16 kHz
* Audio buffer: **1 second**
* Hop size: **100 ms**
* MFCC features + CMVN normalization
* CPU or GPU automatically selected
* `_silence_` and `_unknown_` predictions are ignored

Stop the process using **Ctrl + C**.

---

## How It Works (Pipeline Overview)

1. **Audio capture**

   * Audio is continuously captured from the microphone using `sounddevice`
   * A rolling buffer of length `buffer_size` (default: 1 second) is maintained

2. **Hop-based inference**

   * Every `hop_size` seconds (default: 0.1 s), the current buffer is processed
   * MFCC features are extracted using a fast extraction path
   * Features are normalized using precomputed CMVN statistics
   * The model predicts class probabilities for each hop

3. **Temporal aggregation**

   * Predictions from multiple hops within a 1-second window are aggregated
   * Aggregation strategy is configurable:

     * **Majority vote** over predicted labels
     * **Maximum average confidence** over class probabilities

4. **Post-processing**

   * `_silence_` and `_unknown_` labels are ignored
   * A cooldown mechanism limits repeated prints of the same command

---

## Configurable Options

All realtime parameters live under the `realtime` config group and can be overridden from the command line using Hydra.

### Main Parameters

| Parameter       | Description                                                       |
| --------------- | ----------------------------------------------------------------- |
| `sr`            | Sampling rate used for microphone input                           |
| `buffer_size`   | Length (seconds) of the rolling audio buffer                      |
| `hop_size`      | Step size (seconds) between successive predictions                |
| `cooldown_time` | Minimum time (seconds) before printing the same command again     |
| `selection`     | Aggregation strategy: `majority_vote` or `max_average_confidence` |
| `use_filter`    | Apply denoising before feature extraction                         |
| `filter_type`   | Denoising method: `wiener` or `spectral_subtraction`              |

---

## Changing Configuration from the CLI

You can override any realtime parameter directly from the command line.

### Examples

**Change hop size to 50 ms**

```bash
python main.py pipeline=[realtime] realtime.hop_size=0.05
```

**Increase cooldown to 15 seconds**

```bash
python main.py pipeline=[realtime] realtime.cooldown_time=15
```

**Enable Wiener denoising**

```bash
python main.py pipeline=[realtime] realtime.use_filter=True realtime.filter_type=wiener
```

**Switch aggregation strategy**

```bash
python main.py pipeline=[realtime] realtime.selection=max_average_confidence
```

---

## Cooldown Logic

* A detected command is printed if:

  * Either the cooldown time has elapsed **since the last printed command**
  * Or the newly detected command is **different from the previous one**

This means:

* Repeating the **same command** too quickly is suppressed
* Different commands can still be printed immediately

This behavior helps reduce noisy repeated detections while remaining responsive.

---

## Voting / Aggregation Strategies

Two aggregation methods are implemented:

### 1. Majority Vote

* Each hop produces a predicted label
* The label occurring most frequently over the aggregation window is selected

### 2. Max Average Confidence

* Each hop produces a probability vector
* Probabilities are averaged across hops
* The label with the highest average confidence is selected

Both strategies operate over approximately **1 second of audio**, regardless of hop size (hop count).

```python
if time.time() - last_vote_time >= 1.0:
```

> See `realtime.py` for more detail.

---

## Noise Reduction (Optional)

If enabled, denoising is applied **before feature extraction**:

* Wiener filtering
* Spectral subtraction

These filters use **blind noise estimation** during realtime inference.
Note that classical denoising may not always improve recognition accuracy.

For details and evaluation results, see the
[Noise pipeline](../noise/README.md) and
[Evaluation pipeline](../evaluation/README.md).

---

## Example Console Output

```text
Listening... Ctrl+C to stop
Detected command: up
Detected command: yes
Detected command: left
```

---

## Notes & Limitations

- The realtime pipeline assumes a **model trained on clean speech only**.
  Since noise augmentation is applied only during evaluation, performance may
  degrade in real-world noisy environments.

- Recognition accuracy varies across keywords.
  Short or acoustically similar commands (e.g., `on`, `down`, `right`) may
  require multiple attempts before being reliably detected.

- Currently, the labels `_silence_` and `_unknown_` are **implicitly ignored**
  and never printed when predicted.
  This behavior is hard-coded and will be made configurable in future updates.

- The decision logic is intentionally simple and heuristic-based.
  More advanced temporal smoothing, confidence thresholding, or state-based
  decoding could further improve stability and reduce missed detections.

- Future improvements may include:
  - configurable ignored labels
  - confidence-based rejection thresholds
  - adaptive or label-specific cooldown logic
  - models trained with noise-augmented or denoised data