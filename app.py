from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# load the model from the file
with open('naive_bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get input from the user
    mean_radius = float(request.form['mean_radius'])
    mean_texture = float(request.form['mean_texture'])
    mean_smoothness = float(request.form['mean_smoothness'])

    # make a prediction
    X_new = [[mean_radius, mean_texture, mean_smoothness]]
    Y_pred = model.predict(X_new)

    # return the prediction to the user
    return render_template('result.html', diagnosis=Y_pred[0])

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
