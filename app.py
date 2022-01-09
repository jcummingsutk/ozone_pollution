from flask import Flask, render_template, request

from run import make_prediction

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def diabetes():
    prediction_statement = "please enter some values to obtain a prediction on your diabetes status"
    if request.method == "POST":
        preg = request.form['pregnancies']
        glu = request.form['glucose']
        bp = request.form['blood_pressure']
        ins = request.form['insulin']
        bmi = request.form['bmi']
        ped = request.form['diabetespedigreefunction']
        age = request.form['age']
        pred = make_prediction([[preg, glu, bp, ins, bmi, ped, age]])
        if(pred==0):
            prediction_statement = "The machine learning algorithm has predicted that you do not have diabetes"
        else:
            prediction_statement = "The machine learning algorithm has predicted that you have diabetes"
    return render_template("index.html", prediction=prediction_statement)

@app.route("/sub", methods=["POST"])
def submit():
    if request.method == "POST":
        name = request.form["username"]

    return render_template("sub.html", n=name)

if __name__ == "__main__":
    app.run(debug=True)
