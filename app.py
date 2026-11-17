from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import sys, traceback
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import os, time, threading, hmac
from itsdangerous import URLSafeTimedSerializer

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

SECRET_KEY = os.environ.get('SECRET_KEY', 'change-me-in-render-env')
serializer = URLSafeTimedSerializer(SECRET_KEY)
DEMO_USER = os.environ.get('DEMO_USER', 'demo')
DEMO_PASS = os.environ.get('DEMO_PASS', 'demo123')
PUBLIC_PATHS = {'/api/login', '/api/health', '/api/debug'}

@app.before_request
def require_auth():
    if request.method == 'OPTIONS' or request.path in PUBLIC_PATHS:
        return None
    token = request.headers.get('Authorization', '').replace('Bearer ', '').strip()
    try:
        serializer.loads(token, max_age=8 * 3600)
    except Exception:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    ok = (hmac.compare_digest(str(data.get('username', '')).encode(), DEMO_USER.encode()) and
          hmac.compare_digest(str(data.get('password', '')).encode(), DEMO_PASS.encode()))
    if not ok:
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    return jsonify({'success': True, 'token': serializer.dumps({'u': DEMO_USER}), 'message': 'Login successful'})

@app.route('/api/logout', methods=['POST'])
def logout():
    return jsonify({'success': True})

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"


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


_eval_cache = {}

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:

        data = request.json
        model_type = data.get('model_type', 'lr')

        if model_type in _eval_cache:
            return jsonify({'success': True, 'metrics': _eval_cache[model_type]})
        
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

        _eval_cache[model_type] = metrics
        
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

SCALE_COLS = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
              'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
              'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'MidpointSalary']

X_ALL = df.drop(columns='Diabetes_binary')
X_SAMPLE = X_ALL.sample(n=min(3000, len(X_ALL)), random_state=42).reset_index(drop=True)
X_SAMPLE_SCALED = X_SAMPLE.copy()
X_SAMPLE_SCALED[SCALE_COLS] = scaler.transform(X_SAMPLE[SCALE_COLS])
PCA_COORDS = pca.transform(X_SAMPLE_SCALED)[:, :2]          # computed once
VAR_EXPLAINED = [float(v) for v in pca.explained_variance_ratio_[:2]]

X_SVM_FIT = X_SAMPLE_SCALED.sample(n=min(1000, len(X_SAMPLE_SCALED)), random_state=1)

_anomaly_cache = {}
_key_locks, _key_locks_guard = {}, threading.Lock()

def _lock_for(key):
    with _key_locks_guard:
        return _key_locks.setdefault(key, threading.Lock())

def _mean_diff(frame, mask):
    if not mask.any() or mask.all():
        return {c: 0.0 for c in SCALE_COLS}
    d = frame.loc[mask, SCALE_COLS].mean() - frame.loc[~mask, SCALE_COLS].mean()
    return {c: float(v) for c, v in d.items()}

def _fit_anomaly(model_type, contamination):
    key = (model_type, round(contamination, 3))
    if key in _anomaly_cache:                      # fast path, no lock
        return _anomaly_cache[key], True
    with _lock_for(key):
        if key in _anomaly_cache:
            return _anomaly_cache[key], True
        t0 = time.perf_counter()
        if model_type == 'if':
            mdl = IsolationForest(contamination=contamination, random_state=1, n_estimators=60, n_jobs=1)
            mdl.fit(X_SAMPLE)
            preds = mdl.predict(X_SAMPLE)
        else:
            mdl = OneClassSVM(kernel='rbf', nu=contamination, gamma='scale', cache_size=500)
            mdl.fit(X_SVM_FIT)                     # smaller training subset keeps the fit fast
            preds = mdl.predict(X_SAMPLE_SCALED)
        print(f'[anomaly] {key} took {time.perf_counter() - t0:.1f}s', flush=True)
        anom = preds == -1
        payload = {
            'pca_data': {
                'normal':  {'x': np.round(PCA_COORDS[~anom, 0], 3).tolist(), 'y': np.round(PCA_COORDS[~anom, 1], 3).tolist()},
                'anomaly': {'x': np.round(PCA_COORDS[anom, 0], 3).tolist(),  'y': np.round(PCA_COORDS[anom, 1], 3).tolist()},
            },
            'variance_explained': VAR_EXPLAINED,
            'anomaly_count': int(anom.sum()),
            'total_count': int(len(preds)),
            'feature_differences': _mean_diff(X_SAMPLE, anom),
            'feature_differences_scaled': _mean_diff(X_SAMPLE_SCALED, anom),
            'normal_sample': X_SAMPLE[~anom].head(10).round(3).to_dict('records'),
            'anomaly_sample': X_SAMPLE[anom].head(10).round(3).to_dict('records'),
        }
        _anomaly_cache[key] = (mdl, payload)
        return _anomaly_cache[key], False

def _score_user(mdl, model_type, user_input):
    row = pd.DataFrame([user_input])
    cat = get_bmi_category(float(user_input['BMI']))
    for col, name in (('UnderWeight', 'Underweight'), ('NormalWeight', 'Normal weight'),
                      ('OverWeight', 'Overweight'), ('Obese', 'Obese')):
        row[col] = int(cat == name)
    row = row.reindex(columns=X_ALL.columns, fill_value=0)
    if model_type == 'svm':
        row[SCALE_COLS] = scaler.transform(row[SCALE_COLS])
    return int(mdl.predict(row)[0])

@app.route('/api/anomaly', methods=['POST'])
def detect_anomaly():
    try:
        t0 = time.perf_counter()
        data = request.get_json(silent=True) or {}
        print(f"[anomaly] request received: {data.get('model_type')} {data.get('contamination')}", flush=True)
        model_type = 'svm' if data.get('model_type') in ('svm', 'ocsvm') else 'if'
        contamination = min(max(float(data.get('contamination', 0.05)), 0.005), 0.3)
        (mdl, payload), cached = _fit_anomaly(model_type, contamination)
        user_pred = None
        if data.get('input_data'):
            try:
                user_pred = _score_user(mdl, model_type, data['input_data'])
            except Exception as e:
                print('User scoring failed:', e, flush=True)
        print('[anomaly] sending response', flush=True)
        return jsonify({'success': True, **payload, 'user_prediction': user_pred, 'cached': cached,
                        'elapsed_ms': int((time.perf_counter() - t0) * 1000)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def _warm():
    try:
        _fit_anomaly('if', 0.05)                   # SVM is computed on first use instead
    except Exception as e:
        print('Anomaly warm-up failed:', e, flush=True)
_warm_thread = threading.Thread(target=_warm, daemon=True, name='warmup')
_warm_thread.start()

@app.route('/api/debug', methods=['GET'])
def debug_state():
    frames = sys._current_frames()
    stacks = {}
    for t in threading.enumerate():
        f = frames.get(t.ident)
        stacks[t.name] = traceback.format_stack(f)[-5:] if f else []
    return jsonify({'cache_keys': [list(k) for k in _anomaly_cache.keys()],
                    'warmup_alive': _warm_thread.is_alive(),
                    'threads': stacks})


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': True
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
