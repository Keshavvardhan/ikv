from flask import Flask,render_template,url_for,request,redirect
import numpy as np
import pandas as pd
import joblib
import pickle


app = Flask(__name__)

model = joblib.load('regressor.pkl')


@app.route('/')
@app.route('/main')
def main():
	return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_features =[[x for x in request.form.values()]]
	print(int_features)
	c = ["tv","radio","newspaper"]
	final =pd.DataFrame(int_features,columns=c)
	result = model.predict(final)
	print("The Result is :",result)




	return render_template("main.html",prediction_text=" Expected Sales {}".format(result))


if __name__ == "__main__":
	app.debug=True
	app.run(host = '127.0.0.1', port= 8000)