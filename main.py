import numpy as np
import pickle
from flask import Flask,request,render_template
app=Flask(__name__)
@app.route("/")
def index():
    return render_template ("index.html")

loaded_model = pickle.load(open("model.pkl", "rb"))

 
@app.route('/result', methods = ['POST'])
def result():
      int_features = [int(x) for x in request.form.values()]
      final_features =[np.array(int_features)]
      pre = loaded_model.predict(final_features)
      return render_template("result.html",prediction=pre[0])
if __name__=="__main__":
    app.run(debug=True)