import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import plotly
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)
app.secret_key = 'super_secret_key_churn_predictor'

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# Helper function to load data safely
def load_data():
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return None
    return pd.read_csv(filepath)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            session['filepath'] = filepath
            flash('Dataset uploaded successfully!', 'success')
            return redirect(url_for('preprocess'))
        else:
            flash('Invalid file type. Please upload a CSV.', 'error')
            return redirect(request.url)
    return render_template('upload.html')

@app.route('/preprocess', methods=['GET'])
def preprocess():
    df = load_data()
    if df is None:
        flash('Please upload a dataset first.', 'warning')
        return redirect(url_for('upload'))

    # Drop potential ID columns implicitly (columns with all unique values and categorical/object type)
    cols_to_drop = []
    for col in df.columns:
        if df[col].nunique() == len(df) and df[col].dtype == 'object':
            cols_to_drop.append(col)
        elif col.lower() in ['id', 'customerid', 'customer_id', 'name']:
            cols_to_drop.append(col)
    
    df.drop(columns=list(set(cols_to_drop).intersection(df.columns)), inplace=True)

    # 1. Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # 2. Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                
    # 3. Encode categorical variables
    encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            # Convert to string to avoid mixed type errors before encoding
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
    # Save processed dataframe back to session filepath
    processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_dataset.csv')
    df.to_csv(processed_filepath, index=False)
    session['processed_filepath'] = processed_filepath
    
    # Generate HTML table for preview
    table_html = df.head().to_html(classes='table', index=False)
    
    return render_template('preprocessing.html', 
                           rows=df.shape[0], 
                           cols=df.shape[1], 
                           columns=df.columns.tolist(),
                           table_html=table_html)

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        if 'results' in session and 'best_model_name' in session:
            return render_template('training.html', results=session['results'], best_model=session['best_model_name'])
        else:
            flash('Please train a model first.', 'warning')
            return redirect(url_for('upload'))

    target_col = request.form.get('target_col')
    if not target_col:
        flash('Please select a target column.', 'error')
        return redirect(url_for('preprocess'))
        
    session['target_col'] = target_col
    filepath = session.get('processed_filepath')
    if not filepath or not os.path.exists(filepath):
        flash('Data not found. Please re-upload.', 'error')
        return redirect(url_for('upload'))
        
    df = pd.read_csv(filepath)
    
    if target_col not in df.columns:
        flash(f'Column {target_col} not found in dataset.', 'error')
        return redirect(url_for('preprocess'))

    # Prepare data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Check if target is binary or multiclass
    is_binary = len(y.unique()) == 2
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaler for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler and feature names
    joblib.dump(scaler, os.path.join(app.config['MODEL_FOLDER'], 'scaler.pkl'))
    session['feature_cols'] = X.columns.tolist()
    
    # Initialize models (optimized for low-memory environments like Render free tier)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_accuracy = 0
    best_model_obj = None
    feature_importances = None
    
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            if is_binary:
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if is_binary:
                y_prob = model.predict_proba(X_test)[:, 1]
                
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        
        # Determine average parameter based on binary/multiclass
        avg_method = 'binary' if is_binary else 'macro'
        
        try:
            prec = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
            rec = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
        except Exception:
            prec = 0; rec = 0; f1 = 0
            
        try:
            if is_binary:
                roc_auc = roc_auc_score(y_test, y_prob)
            else:
                # Need predict_proba for all classes for multiclass ROC AUC
                if name == 'Logistic Regression':
                    y_prob_multi = model.predict_proba(X_test_scaled)
                else:
                    y_prob_multi = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_prob_multi, multi_class='ovr')
        except Exception:
            roc_auc = 0.0
            
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'ROC_AUC': roc_auc
        }
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model_obj = model
            if name in ['Decision Tree', 'Random Forest']:
                feature_importances = model.feature_importances_
            else:
                # Try coefficients for LR
                feature_importances = np.abs(model.coef_[0]) if is_binary else np.abs(model.coef_).mean(axis=0)
                
    # Save best model
    joblib.dump(best_model_obj, os.path.join(app.config['MODEL_FOLDER'], 'best_model.pkl'))
    session['best_model_name'] = best_model_name
    
    # Save results to session for dashboard
    session['results'] = results
    
    # Store data for charts
    session['feature_importances'] = feature_importances.tolist() if feature_importances is not None else []
    
    # Store target distribution for dashboard
    target_counts = y.value_counts().to_dict()
    session['target_counts'] = target_counts
    
    return render_template('training.html', results=results, best_model=best_model_name)

@app.route('/dashboard')
def dashboard():
    if 'results' not in session:
        flash('Please train the models first.', 'warning')
        return redirect(url_for('upload'))
        
    results = session['results']
    feature_cols = session.get('feature_cols', [])
    importances = session.get('feature_importances', [])
    target_counts = session.get('target_counts', {})
    
    # 1. Accuracy Chart
    models = list(results.keys())
    accuracies = [results[m]['Accuracy'] for m in models]
    fig_acc = px.bar(x=models, y=accuracies, color=models, labels={'x': 'Model', 'y': 'Accuracy'})
    fig_acc.update_layout(showlegend=False)
    acc_json = json.dumps(fig_acc, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 2. Target Distribution
    fig_dist = px.pie(names=list(target_counts.keys()), values=list(target_counts.values()), hole=0.4)
    dist_json = json.dumps(fig_dist, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 3. Feature Importance
    fig_imp = go.Figure()
    if importances and feature_cols:
        # Sort features
        feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=True).tail(10) # Top 10
        fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h')
    imp_json = json.dumps(fig_imp, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 4. Correlation Heatmap
    corr_json = "{}"
    filepath = session.get('processed_filepath')
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
        corr_matrix = df.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=False, aspect="auto", color_continuous_scale='RdBu_r')
        corr_json = json.dumps(fig_corr, cls=plotly.utils.PlotlyJSONEncoder)
        
    return render_template('dashboard.html', 
                           accuracy_data=acc_json, 
                           target_dist_data=dist_json,
                           feature_imp_data=imp_json,
                           corr_data=corr_json)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'feature_cols' not in session:
        flash('Please train a model first.', 'warning')
        return redirect(url_for('upload'))
        
    features = session['feature_cols']
    prediction = None
    probability = None
    
    if request.method == 'POST':
        try:
            # Gather input data
            input_data = []
            for col in features:
                val = request.form.get(col)
                # Convert to float for prediction
                input_data.append(float(val))
                
            input_array = np.array([input_data])
            
            # Load best model
            model_path = os.path.join(app.config['MODEL_FOLDER'], 'best_model.pkl')
            model = joblib.load(model_path)
            best_model_name = session.get('best_model_name')
            
            # Scale if it was Logistic Regression
            if best_model_name == 'Logistic Regression':
                scaler_path = os.path.join(app.config['MODEL_FOLDER'], 'scaler.pkl')
                scaler = joblib.load(scaler_path)
                input_array = scaler.transform(input_array)
                
            # Predict
            pred_class = model.predict(input_array)[0]
            
            if hasattr(model, "predict_proba"):
                prob_array = model.predict_proba(input_array)[0]
                probability = max(prob_array)
            else:
                probability = 1.0 # fallback
                
            prediction = 'Churn' if pred_class == 1 else 'Stay'
            
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            
    return render_template('prediction.html', features=features, prediction=prediction, probability=probability)

@app.route('/download/<filename>')
def download_file(filename):
    if filename == 'best_model.pkl':
        return send_from_directory(app.config['MODEL_FOLDER'], filename, as_attachment=True)
    elif filename == 'processed_dataset.csv':
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    else:
        flash('File not found.', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
