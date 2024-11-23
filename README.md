# ML-water-potability

# Overview 
This Water Potability Prediction project aims to develop a machine learning model that can predict whether water is safe for consumption or not, based on various water quality features. By analyzing a dataset containing key parameters such as pH, hardness, turbidity, and organic carbon levels, the model classifies water samples as either potable(drinkable) or non-potable(not drinkable). This project employs Python-based data analysis and machine learning techniques to build a predictive model, which is then deployed using Docker for easy distribution and scalability.

## Why This Problem Matters
Access to clean water is a fundamental necessity for human health and well-being. Contaminated water causes diseases and deaths worldwide, especially in developing regions. Machine learning can help detect unsafe water early, which can assist authorities and organizations in taking timely action, ultimately improving public health and quality of life. This project aims to contribute to the broader goal of improving access to clean and safe drinking water by providing an automated solution for water quality assessment.


# Steps
- Data Exploration, visualization and preprocessing
- Training different models, finetuning them, and analyzing their performance, starting with logistic regression
- Decision Tree
- Random Forest
- XGBoost

 # Prerequisites
- Python 3.7 or above
- Docker (if you wish to run the application in a container)

# Libraries Used
- numpy
- pandas
- seaborn
- matplotlib
- scickit-learn
- xgboost
- pickle
- flask
  
# Project Structure
This repository contains:
- **water_potability.csv**: The dataset containing features like pH, hardness, turbidity, and potability (binary label).
- **project.ipynb**: Jupyter Notebook containing the exploratory data analysis (EDA), data preprocessing, different models training, and evaluation steps.
- **train.py**: Script to train the final and best machine learning model.
- **predict.py**: Script for making predictions with the trained model.
- **predict-test-request.py**: Script to test prediction requests.
- **water_model.bin**: The trained model, stored in binary format.
- **load_model_test.py**: A test script to load and validate the model from water_model.bin.
- **Pipfile**: Specifies the Python packages required to run the project.
- **Pipfile.lock**: Contains exact versions of dependencies as installed in the virtual environment.
- **Dockerfile**: The Docker configuration file for containerizing the application.

# Dataset 
Description of the columns:
1. ph: pH of water (0 to 14).
2. Hardness: Capacity of water to precipitate soap in mg/L.
3. Solids: Total dissolved solids in ppm.
4. Chloramines: Amount of Chloramines in ppm.
5. Sulfate: Amount of Sulfates dissolved in mg/L.
6. Conductivity: Electrical conductivity of water in μS/cm.
7. Organic_carbon: Amount of organic carbon in ppm.
8. Trihalomethanes: Amount of Trihalomethanes in μg/L.
9. Turbidity: Water cloudiness, indicating the degree to which light is scattered by particles in the water, in NTU. 
10. Potability: Indicates if water is safe for human consumption. Potable :1 and Not potable :0

# Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/water-potability-prediction.git
   cd water-potability-prediction

2. Set up a virtual environment and install the required dependencies:
   ```bash
    pipenv install

3. (Optional) To run the project inside Docker, build the Docker image:
   ```bash
    docker build -t water-potability-prediction .

4. (Optional) Run the Docker container:
   ```bash
    docker run -p 5000:5000 water-potability-prediction

# Running the Scripts
### Training the Model

To train the machine learning model, run:
```bash
python train.py
````
This script will preprocess the data, train a model, and save it as water_model.bin.

### Making Predictions
Once the model is trained, you can make predictions on new data by running the predict.py script:
```bash
python predict.py
```

You can also test the prediction process by using the predict-test-request.py script.

### Testing the Model
Use load_model_test.py to load and test the model:
```bash
python load_model_test.py
```

# Jupyter Notebook
For a detailed data explorarion and visualization, different model training, parameter tuning, and chosing the final model, open the ``project.ipynb`` notebook. You can run the notebook locally in Jupyter.

# Docker Usage

If you prefer to run the application inside Docker, follow these steps:

- Build the Docker image:
```bash
docker build -t water-potability .
```

### Run the Docker container:

    docker run -p 9696:9696 water-potability

This will start a local server on port 9696, and you can make requests to the application.

# License

This project is licensed under the MIT License - see the LICENSE file for details.


### Final Note

> Thank you for checking out this project! I hope you find it helpful and informative.
> Future work may include incorporating additional features, enhancing model performance, and deploying the project to a cloud service for scalability.
> Contributions are welcome. Feel free to fork this repository, create a pull request, or raise an issue.
> For any questions, feedback, or suggestions, feel free to reach out by [email](serenahaidar77@gmail.com).


