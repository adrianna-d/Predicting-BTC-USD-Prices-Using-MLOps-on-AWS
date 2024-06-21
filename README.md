# Predicting-BTC-USD-Prices-Using-MLOps-on-AWS
Files:

app.py - code for running flask and predictions in postman, it was run on vs code so in case you run it on other programmes add "import sqlite3 " or another db reader. It is not included in the requirements file!

code_for_flask.ipynb - code for app.py but not fully configured

code_model_gru.py -our main code, runs the model, saves the model etc

train.py - trains the mode and saves it in mlflow locally (basically code_model_gru with mlflow)

saved_models - our saved models :D 

requirements.txt - all the pip install you will ever need (done in vs code) 
