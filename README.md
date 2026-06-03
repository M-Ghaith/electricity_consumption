# Electricity Consumption Forecasting

Time series forecasting of electricity consumption using RNN, LSTM, and GRU models in PyTorch.

## Features

- Sequence-based data preprocessing with a sliding window of 96 time steps
- Three recurrent architectures implemented and compared: RNN, LSTM, and GRU
- Trained and evaluated with MSE loss using the Adam optimizer
- Test set evaluation using `torchmetrics` for reproducible scoring

## Tech Stack

- **Python** — core language
- **PyTorch** — neural network models and training
- **pandas / NumPy** — data loading and sequence construction
- **torchmetrics** — evaluation metrics
- **Jupyter Notebook** — experimentation environment

## Quick Start

```bash
# Clone the repo
git clone https://github.com/m-ghaith/electricity_consumption.git
cd electricity_consumption

# Install dependencies
pip install torch torchmetrics pandas numpy jupyter

# Launch the notebook
jupyter notebook electricity_consumption.ipynb
```

## Future Improvements

- [ ] Add hyperparameter tuning (hidden size, number of layers, learning rate)
- [ ] Normalize/scale input features to improve training stability
- [ ] Compare models with a visual plot of predicted vs. actual consumption
