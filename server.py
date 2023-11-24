from flask import Flask, request
import beautyIA
import pandas as pd
app = Flask(__name__)

@app.route("/")
def hello_Word():
    return "<p>Hello world</p>"

@app.route("/predict", methods=['POST'])
def verify():
    data = request.get_json()
    data = data['data']
    response = beautyIA.Predict(data)
    return response

if __name__ == "__main__":
    app.run()