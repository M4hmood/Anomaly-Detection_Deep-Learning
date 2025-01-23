# Anomaly-Detection_Deep-Learning
This is a neural network model (Autoencoder) implemented to detect DDOS (Distributed Denial of Service) anomalies in IOT sensor data (such as blood pressure, heart rate etc...). 

### step 1:
- First, we need to install the required libraries.
- Run the following command in the terminal.
```bash
pip install tqdm torch torchvision pandas numpy
```

### step 2:
- Before training the model, we need to load the data.
- Run the following command in the terminal to load data for the model.
```bash
python utils/data_loader.py
```

### step 3:
- Now, we can train the model.
- Run the following command in the terminal to train the model.
```bash
python training/train.py
```

### step 4:
- Finally, we can test the model.
- Run the following command in the terminal to test the model.
```bash
python inference/run_model.py
```

