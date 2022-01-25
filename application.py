from flask import Flask, render_template, request, jsonify
from main import *
from utils import *

application = Flask(__name__)

@application.route("/", methods=['GET'])
def run_model():
    content_type= request.headers.get('Content-Type')
    print(type(request.get_json()))
    pred = predict(request.get_json())
    print(pred)
    return "The level of ozone is predicted to be {:.3f} ppm".format(pred[0])
    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    #return request.json
    #return jsonify(books)


if __name__ == "__main__":
    application.run(debug=True)
