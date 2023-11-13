import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    expected_order = [ 'clase', 'sexo', 'edad', 'pareja', 'hijos', 'tarifa', 'embarque']
    field_to_column_mapping = {
        'clase': 'Pclass',
        'sexo': {'Mujer': 0, 'Hombre': 1},
        'edad': 'Age',
        'pareja': 'SibSp',
        'hijos': 'Parch',
        'tarifa': 'Fare',
        'embarque': 'Embarked'
    }
    int_features=  [request.form[field_name] for field_name in expected_order]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    predicion_test=""
    if prediction[0]==1 :
            predicion_test="¡Has sobrevivido!"
    else:
            predicion_test="¡Has muerto!"

    return render_template('index.html',prediction_text=predicion_test )

'''
    return render_template('index.html', prediction_text='Sales should be $ {}'.format(prediction))
'''
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)