# CNN Model for Keyword Spotting

This folder contains the **keyword spotting model** implemented in PyTorch. The architecture is lightweight and designed for spectrogram or MFCC features.

---

## Model Architecture

### Convolutional Blocks

The model consists of **3 convolutional blocks**, each containing:

- Conv2D → BatchNorm → ReLU  
- Optional Dropout  
- Optional MaxPooling  

The blocks are designed for inputs of shape `(B, C, F, T)`:

- `B`: batch size  
- `C`: channels (usually 1)  
- `F`: frequency bins  
- `T`: time frames  

### Fully Connected Layers

- Optional **linear hidden layer** (default 64 units)  
- Final linear layer to `num_classes` outputs  
- LogSoftmax activation for classification

**Note:** I used all the 12 commands from the dataset: `"down", "go", "left", "no", "off", "on",
  "right", "stop", "up", "yes", "_silence_", "_unknown_"`. So `num_classes` is 12.

### Global Average Pooling

- Optional global average pooling after convolutional blocks (`use_global_pool`)  
- Reduces feature map to `(B, C)` before FC layers  

---

## Model Hyperparameters

All hyperparameters can be configured via Hydra:

- `in_channels` (default 1)  
- `base_channels` (default 16)  
- `channel_multiplier` (default 2)  
- `dropout_p` (default 0.1)  
- `kernel_size` (default (3,3))  
- `stride` (default (1,1))  
- `padding` (default (1,1))  
- `pool_size` (default (2,2))  
- `use_local_pool` / `use_global_pool`  
- `linear_hidden` (default 64)

---

## Forward Pass

1. Input spectrogram `(B, 1, F, T)`  
2. Pass through 3 convolutional blocks  
3. Apply global average pooling if enabled  
4. Pass through FC layers  
5. Output log-probabilities `(B, num_classes)`  

---

## Weight Initialization

- Conv2D: **Kaiming Normal**  
- BatchNorm2d: weights = 1, bias = 0  
- Linear layers: **Kaiming Uniform**, bias = 0  

---

## References

- [PyTorch Neural Networks](https://pytorch.org/docs/stable/nn.html)

---

> Next: Train the model using the [Training pipeline](../training/README.md)
