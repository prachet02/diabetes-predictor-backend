from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import secrets

app = Flask(__name__)
CORS(app)

# Load all models and scalers
model = joblib.load('diabetes_model.pkl')
features = joblib.load('model_features.pkl')
sc = joblib.load('scaler.pkl')
scaler = joblib.load('xgbscaler.pkl')
dtmodel = joblib.load('diabetes_dtmodel.pkl')
xgbmodel = joblib.load('xgbmodel.pkl')
pca = joblib.load('pca.pkl')

# Load dataset
df = pd.read_csv('processed_diabetes_data.csv')

# Simple session storage
active_sessions = {}

DEMO_USER = 'demo'
DEMO_PASS = 'demo123'


def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if username == DEMO_USER and password == DEMO_PASS:
        session_token = secrets.token_hex(32)
        active_sessions[session_token] = {'username': username}
        
        return jsonify({
            'success': True,
            'token': session_token,
            'message': 'Login successful'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Invalid credentials'
        }), 401


@app.route('/api/logout', methods=['POST'])
def logout():
    token = request.headers.get('Authorization')
    if token and token in active_sessions:
        del active_sessions[token]
    return jsonify({'success': True, 'message': 'Logged out successfully'})


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_type = data.get('model_type', 'lr')
        user_input = data.get('input_data')
        
        input_df = pd.DataFrame([user_input])
        
        bmi = user_input['BMI']
        bmi_category = get_bmi_category(bmi)
        
        input_df['UnderWeight'] = 1 if bmi_category == "Underweight" else 0
        input_df['NormalWeight'] = 1 if bmi_category == "Normal weight" else 0
        input_df['OverWeight'] = 1 if bmi_category == "Overweight" else 0
        input_df['Obese'] = 1 if bmi_category == "Obese" else 0
        
        input_df = input_df[features]
        
        cols = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
                'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
                'MidpointSalary']
        
        prediction_proba = None
        prediction_class = None
        
        if model_type == 'lr':
            input_df[cols] = sc.transform(input_df[cols])
            prediction_proba = float(model.predict_proba(input_df)[0, 1])
            prediction_class = int(model.predict(input_df)[0])
            
        elif model_type == 'xgb':
            input_df[cols] = scaler.transform(input_df[cols])
            prediction_proba = float(xgbmodel.predict_proba(input_df)[0, 1])
            prediction_class = int(xgbmodel.predict(input_df)[0])
            
        elif model_type == 'dt':
            input_df[cols] = sc.transform(input_df[cols])
            prediction_class = int(dtmodel.predict(input_df)[0])
            prediction_proba = None
        
        feature_importances = xgbmodel.feature_importances_
        feature_names = features
        
        importance_dict = {
            name: float(importance) 
            for name, importance in zip(feature_names, feature_importances)
        }
        
        sorted_importance = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return jsonify({
            'success': True,
            'prediction_class': prediction_class,
            'prediction_proba': prediction_proba,
            'bmi_category': bmi_category,
            'feature_importance': sorted_importance
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.json
        model_type = data.get('model_type', 'lr')
        
        cols = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
                'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
                'MidpointSalary']
        
        transformed_dataset = df.copy()
        
        if model_type == 'xgb':
            transformed_dataset[cols] = scaler.transform(transformed_dataset[cols])
        else:
            transformed_dataset[cols] = sc.transform(transformed_dataset[cols])
        
        X = transformed_dataset.iloc[:, 1:]
        y = transformed_dataset.iloc[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1
        )
        
        if model_type == 'lr':
            y_pred = model.predict(X_test)
        elif model_type == 'dt':
            y_pred = dtmodel.predict(X_test)
        elif model_type == 'xgb':
            y_pred = xgbmodel.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'confusion_matrix': {
                'tn': int(cm[0, 0]),
                'fp': int(cm[0, 1]),
                'fn': int(cm[1, 0]),
                'tp': int(cm[1, 1])
            }
        }
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/explore', methods=['POST'])
def explore():
    try:
        data = request.json
        filters = data.get('filters', {})
        
        df_explore = df.copy()
        df_explore['BMI_Category'] = df_explore['BMI'].apply(get_bmi_category)
        df_explore['Sex_Label'] = df_explore['Sex'].map({0: 'Female', 1: 'Male'})
        df_explore['Diabetes_Label'] = df_explore['Diabetes_binary'].map(
            {0: 'No Diabetes', 1: 'Diabetes'}
        )
        
        # Apply filters
        filtered_df = df_explore.copy()
        
        # Categorical filters
        if filters.get('HighBP') and filters['HighBP'] != 'All':
            filtered_df = filtered_df[filtered_df['HighBP'] == int(filters['HighBP'])]
        if filters.get('HighChol') and filters['HighChol'] != 'All':
            filtered_df = filtered_df[filtered_df['HighChol'] == int(filters['HighChol'])]
        if filters.get('Smoker') and filters['Smoker'] != 'All':
            filtered_df = filtered_df[filtered_df['Smoker'] == int(filters['Smoker'])]
        if filters.get('Sex') and filters['Sex'] != 'All':
            filtered_df = filtered_df[filtered_df['Sex'] == int(filters['Sex'])]
        if filters.get('Fruits') and filters['Fruits'] != 'All':
            filtered_df = filtered_df[filtered_df['Fruits'] == int(filters['Fruits'])]
        if filters.get('Veggies') and filters['Veggies'] != 'All':
            filtered_df = filtered_df[filtered_df['Veggies'] == int(filters['Veggies'])]
        if filters.get('PhysActivity') and filters['PhysActivity'] != 'All':
            filtered_df = filtered_df[filtered_df['PhysActivity'] == int(filters['PhysActivity'])]
        if filters.get('HvyAlcoholConsump') and filters['HvyAlcoholConsump'] != 'All':
            filtered_df = filtered_df[filtered_df['HvyAlcoholConsump'] == int(filters['HvyAlcoholConsump'])]
        if filters.get('DiffWalk') and filters['DiffWalk'] != 'All':
            filtered_df = filtered_df[filtered_df['DiffWalk'] == int(filters['DiffWalk'])]
        if filters.get('AnyHealthcare') and filters['AnyHealthcare'] != 'All':
            filtered_df = filtered_df[filtered_df['AnyHealthcare'] == int(filters['AnyHealthcare'])]
        if filters.get('HeartDiseaseorAttack') and filters['HeartDiseaseorAttack'] != 'All':
            filtered_df = filtered_df[filtered_df['HeartDiseaseorAttack'] == int(filters['HeartDiseaseorAttack'])]
        if filters.get('Stroke') and filters['Stroke'] != 'All':
            filtered_df = filtered_df[filtered_df['Stroke'] == int(filters['Stroke'])]
        
        # Range filters
        if filters.get('BMI_min'):
            filtered_df = filtered_df[filtered_df['BMI'] >= float(filters['BMI_min'])]
        if filters.get('BMI_max'):
            filtered_df = filtered_df[filtered_df['BMI'] <= float(filters['BMI_max'])]
        if filters.get('Age_min'):
            filtered_df = filtered_df[filtered_df['Age'] >= int(filters['Age_min'])]
        if filters.get('Age_max'):
            filtered_df = filtered_df[filtered_df['Age'] <= int(filters['Age_max'])]
        if filters.get('GenHlth_min'):
            filtered_df = filtered_df[filtered_df['GenHlth'] >= int(filters['GenHlth_min'])]
        if filters.get('GenHlth_max'):
            filtered_df = filtered_df[filtered_df['GenHlth'] <= int(filters['GenHlth_max'])]
        
        # Diabetes by sex counts
        diabetes_sex_counts = filtered_df.groupby(['Diabetes_Label', 'Sex_Label']).size().to_dict()
        
        # BMI distribution
        bmi_bins = list(range(10, 65, 2))
        bmi_hist, bmi_edges = np.histogram(filtered_df['BMI'], bins=bmi_bins)
        bmi_distribution = {str(int(edge)): int(count) for edge, count in zip(bmi_edges[:-1], bmi_hist)}
        
        # Age distribution
        age_counts = filtered_df['Age'].value_counts().sort_index().to_dict()
        age_distribution = {str(k): int(v) for k, v in age_counts.items()}
        
        # Salary distribution
        salary_counts = filtered_df['MidpointSalary'].value_counts().sort_index().to_dict()
        salary_distribution = {str(k): int(v) for k, v in salary_counts.items()}
        
        # Get sample of filtered data (first 10 rows)
        sample_data = filtered_df.head(10).to_dict('records')
        
        return jsonify({
            'success': True,
            'data': {
                'diabetes_by_sex': {str(k): int(v) for k, v in diabetes_sex_counts.items()},
                'bmi_distribution': bmi_distribution,
                'age_distribution': age_distribution,
                'salary_distribution': salary_distribution,
                'total_records': len(filtered_df),
                'sample_data': sample_data
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/anomaly', methods=['POST'])
def detect_anomaly():
    try:
        data = request.json
        model_type = data.get('model_type', 'if')
        contamination = data.get('contamination', 0.05)
        user_input = data.get('input_data')
        
        cols = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
                'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
                'MidpointSalary']
        
        # Sample data for faster processing (use 5000 samples instead of all)
        X = df.drop('Diabetes_binary', axis=1).sample(n=min(5000, len(df)), random_state=42)
        X_original = X.copy()
        scaled_df = X.copy()
        scaled_df[cols] = scaler.transform(scaled_df[cols])
        
        if model_type == 'if':
            anomaly_model = IsolationForest(
                contamination=contamination, 
                random_state=1, 
                n_estimators=100
            )
            anomaly_model.fit(X)
            anomaly_scores = anomaly_model.decision_function(X)
            anomaly_predictions = anomaly_model.predict(X)
        else:
            anomaly_model = OneClassSVM(
                kernel='rbf', 
                nu=contamination, 
                gamma='scale'
            )
            anomaly_model.fit(scaled_df)
            anomaly_scores = anomaly_model.decision_function(scaled_df)
            anomaly_predictions = anomaly_model.predict(scaled_df)
        
        comps = pca.transform(scaled_df)
        pca_data = pd.DataFrame(data=comps, columns=['PC1', 'PC2'])
        pca_data['anomaly'] = anomaly_predictions
        pca_data['anomaly_score'] = anomaly_scores
        
        normal_indices = pca_data[pca_data['anomaly'] == 1].index.tolist()
        anomaly_indices = pca_data[pca_data['anomaly'] == -1].index.tolist()
        
        user_prediction = None
        if user_input:
            input_df = pd.DataFrame([user_input])
            
            if model_type == 'if':
                user_prediction = int(anomaly_model.predict(input_df)[0])
            else:
                input_df[cols] = scaler.transform(input_df[cols])
                user_prediction = int(anomaly_model.predict(input_df)[0])
        
        # Get unscaled data
        normal_data_unscaled = X_original.iloc[normal_indices]
        anomaly_data_unscaled = X_original.iloc[anomaly_indices]
        
        # Get scaled data
        normal_data_scaled = scaled_df.iloc[normal_indices]
        anomaly_data_scaled = scaled_df.iloc[anomaly_indices]
        
        # Calculate mean differences (unscaled)
        mean_normal_unscaled = normal_data_unscaled[cols].mean().to_dict()
        mean_anomaly_unscaled = anomaly_data_unscaled[cols].mean().to_dict()
        
        feature_differences_unscaled = {
            feature: float(mean_anomaly_unscaled[feature] - mean_normal_unscaled[feature])
            for feature in cols
        }
        
        # Calculate mean differences (scaled)
        mean_normal_scaled = normal_data_scaled[cols].mean().to_dict()
        mean_anomaly_scaled = anomaly_data_scaled[cols].mean().to_dict()
        
        feature_differences_scaled = {
            feature: float(mean_anomaly_scaled[feature] - mean_normal_scaled[feature])
            for feature in cols
        }
        
        # Sample datasets for display
        normal_sample = normal_data_unscaled.head(10).to_dict('records')
        anomaly_sample = anomaly_data_unscaled.head(10).to_dict('records')
        
        return jsonify({
            'success': True,
            'pca_data': {
                'normal': pca_data[pca_data['anomaly'] == 1][['PC1', 'PC2']].to_dict('records'),
                'anomaly': pca_data[pca_data['anomaly'] == -1][['PC1', 'PC2']].to_dict('records')
            },
            'variance_explained': [float(pca.explained_variance_ratio_[0]), 
                                  float(pca.explained_variance_ratio_[1])],
            'anomaly_count': len(anomaly_indices),
            'total_count': len(pca_data),
            'user_prediction': user_prediction,
            'feature_differences': feature_differences_unscaled,
            'feature_differences_scaled': feature_differences_scaled,
            'normal_sample': normal_sample,
            'anomaly_sample': anomaly_sample
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': True
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
