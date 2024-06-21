# Predicting-BTC-USD-Prices-Using-MLOps-on-AWS

## Table of contents
* [Introduction](#Intro)
* [Installation](#Installation)
* [Files](#Files)


## Intro

## Installation

Project is created with:
* Python 3.7

```
$ pip install ...

```
	
### start
To run this project, run the code_model_gru.py file, and then app.py 

```
$ python code_model_gru.py
$ python app.py
locally for Postman: http://127.0.0.1:5000/predict

## Files:

app.py - code for running flask and predictions in postman, it was run on vs code so in case you run it on other programmes add "import sqlite3 " or another db reader. It is not included in the requirements file!

code_for_flask.ipynb - code for app.py but not fully configured

code_model_gru.py -our main code, runs the model, saves the model etc

train.py - trains the mode and saves it in mlflow locally (basically code_model_gru with mlflow)

saved_models - our saved models :D 

requirements.txt - all the pip install you will ever need (done in vs code) 
<<<<<<< HEAD

crypto_predictions.db - database that stores predictions and last 30 entries for each data fetch 
=======
>>>>>>> e78c380b738735b752573a19a1a8a5f57f470fa9
