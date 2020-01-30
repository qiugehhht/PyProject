from flask import Flask, render_template, request, redirect, url_for
from model import read_data, decision_tree_algo, log_regression_algo, heatplt, pre_random_m, random_tree_algo
import pandas as pd
import numpy as np
import os
from flask_bootstrap import Bootstrap


app = Flask(__name__)
Bootstrap(app)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY


# Show preprocessed data as a table
@app.route('/index', methods = ("POST", "GET") )
def data_table():
   data = read_data() # Shows only 50 samples
   return render_template('index.html',  tables=[data.to_html(classes='table table-striped')])

# Run a grid search of decision tree and get the best estimator
@app.route('/decision', methods = ("POST", "GET") )
def decision_tree_view():
   s = decision_tree_algo()
   dresult = s.best_estimator_
   return render_template('decision.html', result=dresult)

# Run a grid search of random forest and get the best estimator
@app.route('/random', methods = ("POST", "GET") )
def random_tree_view():
   s = random_tree_algo()
   rresult = s.best_estimator_
   return render_template('random.html', result=rresult)

# Get score of logistic regression
@app.route('/log', methods = ("POST", "GET") )
def log_regression_view():
   lresult = log_regression_algo()
   iresult = str(lresult)
   return iresult

# Get a image of data visualization (heatmap)
@app.route('/plt', methods = ("POST", "GET") )
def heat_plt_view():
   s = heatplt()
   figure = s.get_figure()    
   figure.savefig(os.path.join('static', 'heat.png'), dpi = 400)
   return render_template('plt.html')



# Submit a form of user defined parameters and redirect to "pre_random.html" to get the accuracy of this model
@app.route('/random_tree_pre', methods = ("POST","GET"))
def post_random_tree_pre():
   if request.method == "POST":
      ntrees = request.form['ntrees']
      mtry = request.form['mtry']
      depth = request.form['depth']
      return redirect(url_for("pre_random", ntrees = ntrees, mtry = mtry, depth = depth))
   return render_template("random_tree_pre.html")

# Shows accuracy of user defined model, and provide form to receive ueser defined variables and make prediction
@app.route('/pre_random/<int:ntrees>/<int:mtry>/<int:depth>', methods = ("POST", "GET") )
def pre_random(ntrees, mtry, depth):
      s = pre_random_m(ntrees,mtry,depth)
      sss = s[0]
      if request.method == "POST":
         int_features = [float(x) for x in request.form.values()]
         final_features = [np.array(int_features)]
         prediction = s[1].predict(final_features)
         return render_template('pre_random.html', result=sss, prediction = prediction)
      return render_template('pre_random.html', result=sss)

if __name__ == '__main__':
   app.run(debug = True, port = 5000)