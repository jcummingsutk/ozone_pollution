from flask import Flask, render_template, request,jsonify
from main import *
from utils import *

app = Flask(__name__)

books = [
    {'id': 0,
     'title': 'A Fire Upon the Deep',
     'author': 'Vernor Vinge',
     'first_sentence': 'The coldsleep itself was dreamless.',
     'year_published': '1992'},
    {'id': 1,
     'title': 'The Ones Who Walk Away From Omelas',
     'author': 'Ursula K. Le Guin',
     'first_sentence': 'With a clamor of bells that set the swallows soaring, the Festival of Summer came to the city Omelas, bright-towered by the sea.',
     'published': '1973'},
    {'id': 2,
     'title': 'Dhalgren',
     'author': 'Samuel R. Delany',
     'first_sentence': 'to wound the autumnal city.',
     'published': '1975'}
]

@app.route("/test", methods=['GET'])
def api_filter_json():
    content_type= request.headers.get('Content-Type')
    print(type(request.get_json()))
    pred = predict(request.get_json())
    print(pred)
    return "The level of ozone is predicted to be {:.3f} ppm".format(pred[0])
    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    #return request.json
    #return jsonify(books)

@app.route('/hello/', methods=['GET','POST'])
def welcome():
    return "hello you"

@app.route("/", methods=["GET", "POST"])
def diabetes():
    prediction_statement = (
        "please enter some values to obtain a prediction on your diabetes status"
    )
    if request.method == "POST":
        preg = request.form["pregnancies"]
        glu = request.form["glucose"]
        bp = request.form["blood_pressure"]
        ins = request.form["insulin"]
        bmi = request.form["bmi"]
        ped = request.form["diabetespedigreefunction"]
        age = request.form["age"]
        pred = make_prediction([[preg, glu, bp, ins, bmi, ped, age]])
        if pred == 0:
            prediction_statement = "The machine learning algorithm has predicted that you do not have diabetes"
        else:
            prediction_statement = (
                "The machine learning algorithm has predicted that you have diabetes"
            )
    return render_template("index.html", prediction=prediction_statement)


@app.route("/sub", methods=["POST"])
def submit():
    if request.method == "POST":
        name = request.form["username"]

    return render_template("sub.html", n=name)


if __name__ == "__main__":
    app.run(debug=True)
