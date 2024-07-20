# Lottery Prediction with Machine Learning 

This repository contains two projects that use machine learning models to predict future lottery numbers for Mega Millions and Powerball.

## Project Structure

* **megamillions.csv:**  Dataset containing historical Mega Millions draw data.
* **powerball.csv:** Dataset containing historical Powerball draw data.
* **predict_megamillions.ipynb:** Jupyter Notebook demonstrating the prediction model for Mega Millions.
* **predict_powerball.ipynb:** Jupyter Notebook demonstrating the prediction model for Powerball.

## Project Overview

### Data Preprocessing

1. **Data Loading:** The datasets are loaded using Pandas.
2. **Feature Selection:** Relevant features (lottery numbers) are selected.
3. **Data Splitting:** The data is split into training sets and labels.
4. **Normalization:**  The data is standardized using `StandardScaler`.

### Model Architecture

* **LSTM-Based Neural Network:** A bidirectional LSTM (Long Short-Term Memory) network is used for both projects. 
* **Hyperparameter Tuning:** The model architecture and learning parameters are tuned for each lottery type.
* **Training:** The models are trained on historical data to learn patterns and trends.

### Prediction

* **Rolling Window Approach:** A sliding window is used to create sequences of past numbers for prediction.
* **Model Inference:** The trained model predicts the next set of lottery numbers.
* **Post-Processing:** The predicted numbers are denormalized and presented in a human-readable format.

## Usage

1. **Install dependencies:** Ensure you have Python, Jupyter Notebook, and the following libraries installed:
   - NumPy
   - Pandas
   - scikit-learn
   - TensorFlow (or Keras)

2. **Open the Jupyter Notebooks:**
   - `predict_megamillions.ipynb`
   - `predict_powerball.ipynb`

3. **Run the cells:** Execute the cells in the notebooks sequentially to load the data, train the models, and generate predictions.

## Results

The models provide predictions for the next possible winning numbers in each lottery. Due to the inherent randomness of lotteries, these predictions are not guaranteed, but they offer insights based on historical trends.

## Disclaimer

This project is for educational and informational purposes only. The lottery is a game of chance, and there is no guarantee that these predictions will be accurate. 

## Contributing

Feel free to fork this repository and experiment with different models, data sources, or features to improve the prediction accuracy. Contributions are welcome!

## License

This project is licensed under the MIT License.
