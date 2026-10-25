# Neural Network & Deep Learning - Task 2

A comprehensive neural network implementation with backpropagation for multi-class classification, featuring a modern GUI and automated hyperparameter grid search.

## Features

- **Custom Neural Network Implementation**
  - Multi-layer feedforward neural network with configurable architecture
  - Backpropagation training algorithm
  - Support for multiple activation functions (Sigmoid, Tanh)
  - Optional bias terms in hidden and output layers

- **Interactive GUI**
  - Dark-themed modern interface built with Tkinter
  - Three main tabs: Training, Experiments, and Classification
  - Real-time training progress visualization
  - Confusion matrix display for model evaluation

- **Automated Hyperparameter Grid Search**
  - Quick and Full grid search modes
  - Automated testing of multiple hyperparameter combinations
  - Results displayed in sortable tables
  - Best results highlighted per activation function
  - CSV export functionality for detailed analysis

- **Data Preprocessing Pipeline**
  - Automatic missing value imputation
  - Feature normalization (z-score)
  - One-hot encoding for categorical targets
  - Support for single sample classification

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Neural_Network&Deep_learning_Task2
```

2. Install required dependencies:
```bash
pip install numpy pandas
```

3. Ensure the dataset is in place:
- Place `penguins.csv` in the `data/` directory

## Usage

### Running the Application

Launch the GUI by running:
```bash
python main.py
```

### Training Tab

1. **Configure Model Parameters**
   - Number of hidden layers (1-10)
   - Neurons per layer (comma-separated, e.g., "4,5")
   - Learning rate (η)
   - Number of epochs
   - Bias toggle (enabled/disabled)
   - Activation function (Sigmoid or Tanh)

2. **Train the Model**
   - Click "Train Model" to start training
   - Monitor progress with the progress bar
   - View training and test accuracy
   - Examine the confusion matrix

### Experiments Tab

1. **Select Grid Search Mode**
   - **Quick Grid**: Fast check with limited combinations
   - **Full Grid**: Comprehensive search with all combinations

2. **Run Grid Search**
   - Click "Run Grid Search" to begin
   - Monitor progress and completion status
   - View all results in the sortable table
   - Best results per activation function are highlighted

3. **Save Results**
   - Click "Save CSV" to export results
   - Files saved to `results/` directory:
     - `all_runs.csv`: All grid search configurations
     - `best.csv`: Best result per activation function
     - `best_summary.txt`: Formatted summary with confusion matrices

### Classification Tab

1. **Train a Model First**
   - Complete training in the Training tab to enable classification

2. **Enter Sample Features**
   - Input numeric feature values
   - Select origin location from dropdown
   - Click "Classify" to predict class

## Project Structure

```
Neural_Network&Deep_learning_Task2/
├── data/
│   └── penguins.csv              # Dataset
├── logic/
│   ├── activations.py            # Activation functions (Sigmoid, Tanh)
│   ├── data_loader.py            # Data loading and splitting
│   ├── metrics.py                # Evaluation metrics (accuracy, confusion matrix)
│   ├── neural_network.py         # Neural network implementation
│   ├── pipeline.py               # Main pipeline orchestrator
│   └── preprocessing.py          # Data preprocessing utilities
├── ui/
│   └── app.py                    # GUI application
├── experiments.py                # Hyperparameter grid search
├── main.py                       # Entry point
├── requirements                  # Python dependencies
└── README.md                     # This file
```

## Model Architecture

The neural network supports:
- **Input Layer**: Size matches number of features
- **Hidden Layers**: Configurable number and size
- **Output Layer**: Size matches number of classes (one-hot encoding)
- **Activation Functions**: Sigmoid or Tanh for hidden layers
- **Bias Terms**: Optional bias in all layers

## Dataset

The project uses the Penguins dataset for multi-class classification:
- **Features**: Numeric measurements and origin location
- **Target**: Species classification
- **Preprocessing**: Missing value imputation, normalization, encoding

## Dependencies

- Python 3.12
- NumPy
- Pandas
- Tkinter (usually included with Python)

## License

This project is part of Task 2 for Neural Networks & Deep Learning course.

## Author

Developed as part of the Neural Networks & Deep Learning course assignment.
