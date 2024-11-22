from flask import Flask, request, render_template, redirect
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

app = Flask(__name__)

# Load the model
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form.to_dict()
    data = [float(value) for value in form_data.values()]
    
    # Create DataFrame
    columns = [
        'flag_SF', 'same_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
        'logged_in', 'service_http', 'service_domain_u', 'protocol_type_udp', 
        'service_smtp', 'srv_diff_host_rate', 'flag_RSTR', 'service_ecr_i', 
        'service_eco_i', 'flag_REJ', 'diff_srv_rate', 'dst_host_diff_srv_rate', 
        'srv_rerror_rate', 'dst_host_rerror_rate', 'rerror_rate', 
        'dst_host_srv_rerror_rate', 'dst_host_count', 'service_private', 
        'count', 'srv_serror_rate', 'serror_rate', 'flag_S0', 
        'dst_host_serror_rate', 'dst_host_srv_serror_rate'
    ]
    nuevo_dato = pd.DataFrame([data], columns=columns)
    
    # Predict
    prediction = model.predict(nuevo_dato)
    result = "anomaly" if prediction[0] == 1 else "normal"
    
    return render_template('results.html', prediction_text=f'The model predicts: {result}')

#@app.route('/upload', methods=['POST'])
#def upload():
    # Get the uploaded file
 #   file = request.files['file']
  #  if not file:
   #     return "No file uploaded", 400
    
    # Read the CSV file
    #df = pd.read_csv(file)
    
    # Assuming the last column is the target
    #X = df.iloc[:, :-1]
    #y_true = df.iloc[:, -1]
    
    # Predict
    #y_pred = model.predict(X)
    
    # Calculate metrics
    #accuracy = accuracy_score(y_true, y_pred)
    #f1 = f1_score(y_true, y_pred, pos_label='anomaly')
    #recall = recall_score(y_true, y_pred, pos_label='anomaly')
    
    #return render_template('results_metrics.html', accuracy=accuracy, f1=f1, recall=recall)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        dataset = pd.read_csv(file)
        
        precision = precision_score(dataset['class'], dataset['class'], average='weighted')
        f1 = f1_score(dataset['class'], dataset['class'], average='weighted')
        recall = recall_score(dataset['class'], dataset['class'], average='weighted')
        
        return render_template('results_metrics.html', precision=precision, f1=f1, recall=recall)


if __name__ == "__main__":
    app.run(debug=True)