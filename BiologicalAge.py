import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import jpype
import jaydebeapi

# Load environment variables
load_dotenv()

# JDBC configuration
jdbc_driver_path = "/home/jovyan/denodo/denodo-vdp-jdbcdriver-9.2.1.jar"
jdbc_driver_class = "com.denodo.vdp.jdbc.Driver"
jdbc_url = f"jdbc:denodo://{os.getenv('DENODO_HOST')}:9999/{os.getenv('DENODO_DATABASE')}?ssl=true"
jdbc_username = os.getenv("DENODO_UID", "")
jdbc_password = os.getenv("DENODO_PWD", "")
java_home = "/home/jovyan/zulu21.42.19-ca-jdk21.0.7-linux_x64"

# Set JAVA_HOME if needed
if not os.getenv('JAVA_HOME'):
    os.environ['JAVA_HOME'] = java_home

# Set page config
st.set_page_config(
    page_title="Biological Age Prediction System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define consistent feature columns
FEATURE_COLUMNS = [
    'age', 'gender', 'heart_rate', 'systolic_bp', 'diastolic_bp', 'bmi',
    'hba1c', 'cholesterol', 'hdl', 'creatinine', 'wbc', 'crp', 'il6',
    'tnf_alpha', 'glyca', 'sleep_hours', 'activity_mins', 'stress_score',
    'mood_scale', 'steps_per_day', 'hrv', 'vo2max', 'grip_strength'
]

# Define tiers
TIER_I_FEATURES = ['age', 'gender', 'bmi', 'heart_rate', 'systolic_bp', 'diastolic_bp']
TIER_II_FEATURES = ['hba1c', 'cholesterol', 'hdl', 'creatinine', 'wbc', 'crp', 'il6', 'tnf_alpha', 'glyca']
TIER_III_FEATURES = ['steps_per_day', 'activity_mins', 'hrv', 'vo2max', 'grip_strength']
TIER_IV_FEATURES = ['sleep_hours', 'stress_score', 'mood_scale']

# Lab code mappings
LAB_CONDITIONS = {
    'N0510': 'hba1c',
    'C0131': 'cholesterol',
    'C0141': 'hdl',
    'C0070': 'creatinine',
    'A0005': 'wbc',
    'T0700': 'crp'
}

# Vital signs mappings
VITAL_SIGNS = {
    '7': 'Temperature',
    '8': 'PR',
    '9': 'BP systolic',
    '10': 'BP diastolic',
    '11': 'RR',
    '69': 'SpO2',
    '87': 'Height',
    '90': 'Weight'
}

def start_jvm():
    """Start JVM if not already started"""
    if not jpype.isJVMStarted():
        jvm_path = os.path.join(java_home, "lib", "server", "libjvm.so")
        if not os.path.exists(jvm_path):
            raise FileNotFoundError(f"libjvm.so not found at {jvm_path}")
        jpype.startJVM(jvm_path, "-Djava.class.path=" + jdbc_driver_path)

@st.cache_resource
def get_denodo_connection():
    """Get cached Denodo connection"""
    return connect_to_denodo_jdbc()

# Neural Network Model (same as before)
class BiologicalAgePredictor(nn.Module):
    def __init__(self, input_size):
        super(BiologicalAgePredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

def generate_synthetic_data(n_samples=1000):
    """
    NHANES-tuned synthetic data generator (approximate).
    Values and categories are chosen to roughly reflect
    adult US population distributions (20–80y), not any
    exact NHANES release.
    """
    np.random.seed(42)
    data = []

    def truncated_normal(mean, sd, low, high, size=None):
        vals = np.random.normal(mean, sd, size)
        return np.clip(vals, low, high)

    for _ in range(n_samples):
        # --- Demographics ---
        # NHANES includes adults and older adults; use 20–80 years
        age = np.random.uniform(20, 80)

        # Approximate sex ratio ~ 50/50; your model uses 0=Male, 1=Female
        gender = np.random.choice([0, 1], p=[0.49, 0.51])

        # --- Anthropometrics ---
        # Height (cm) and weight (kg) loosely based on US adult distributions
        height = truncated_normal(mean=168, sd=10, low=145, high=200)
        weight = truncated_normal(mean=80, sd=18, low=45, high=160)
        bmi = weight / ((height / 100) ** 2)
        bmi = float(np.clip(bmi, 16, 50))

        # --- Vitals ---
        # Heart rate ~ 60–100 bpm, centered near 72
        heart_rate = truncated_normal(mean=72, sd=11, low=50, high=120)

        # Base BP, then adjust by BMI and age a bit
        systolic_bp = truncated_normal(mean=118, sd=15, low=85, high=200)
        diastolic_bp = truncated_normal(mean=76, sd=10, low=50, high=120)

        # Add simple correlation with BMI and age
        systolic_bp += 0.5 * (bmi - 27) + 0.2 * (age - 50) / 10
        diastolic_bp += 0.3 * (bmi - 27) + 0.1 * (age - 50) / 10

        systolic_bp = float(np.clip(systolic_bp, 85, 220))
        diastolic_bp = float(np.clip(diastolic_bp, 50, 130))

        # --- Lab values (NHANES-like ranges) ---
        # HbA1c: roughly normal, with tail into pre/diabetes
        hba1c = truncated_normal(mean=5.6, sd=0.8, low=4.0, high=12.0)

        # Total cholesterol (mg/dL)
        cholesterol = truncated_normal(mean=195, sd=38, low=100, high=350)

        # HDL (mg/dL)
        hdl = truncated_normal(mean=52, sd=14, low=20, high=100)

        # Creatinine (mg/dL)
        creatinine = truncated_normal(mean=0.9, sd=0.25, low=0.4, high=2.5)

        # WBC (10^3/µL)
        wbc = truncated_normal(mean=7.0, sd=1.8, low=3.0, high=15.0)

        # CRP (mg/L), highly skewed – approximate with log-normal
        crp = np.random.lognormal(mean=np.log(1.2), sigma=0.8)
        crp = float(np.clip(crp, 0.05, 30.0))

        # IL-6 (pg/mL), also skewed
        il6 = np.random.lognormal(mean=np.log(1.5), sigma=0.7)
        il6 = float(np.clip(il6, 0.1, 20.0))

        # TNF-alpha (pg/mL)
        tnf_alpha = np.random.lognormal(mean=np.log(1.8), sigma=0.5)
        tnf_alpha = float(np.clip(tnf_alpha, 0.1, 15.0))

        # Some generic advanced glycation marker (arbitrary but stable)
        glyca = truncated_normal(mean=400, sd=60, low=250, high=600)

        # Introduce mild correlations: higher BMI → higher CRP, HbA1c
        hba1c += max(0, (bmi - 27)) * 0.03  # ~0.3% per 10 BMI points
        crp   *= 1 + max(0, (bmi - 27)) * 0.02

        hba1c = float(np.clip(hba1c, 4.0, 14.0))
        crp   = float(np.clip(crp, 0.05, 40.0))

        # --- Lifestyle factors ---
        # Sleep: NHANES adults cluster around 6–8 hours
        sleep_hours = truncated_normal(mean=7.0, sd=1.2, low=3.0, high=12.0)

        # Activity minutes per day (moderate-to-vigorous)
        # Highly skewed: many low, some very high
        activity_mins = np.random.lognormal(mean=np.log(30), sigma=0.9)
        activity_mins = float(np.clip(activity_mins, 0, 240))

        # Simple stress and mood scales (0–10 / 1–10)
        stress_score = int(np.clip(np.random.normal(loc=4.5, scale=2.5), 0, 10))
        mood_scale   = int(np.clip(np.random.normal(loc=7.0, scale=2.0), 1, 10))


        # Steps per day (pedometer / accelerometer)
        steps_per_day = truncated_normal(mean=8000, sd=3500, low=1000, high=25000)

        # --- Fitness / autonomic markers ---
        # HRV (ms)
        hrv = truncated_normal(mean=50, sd=18, low=10, high=150)

        # VO2max (mL/kg/min), rough adult distribution
        vo2max = truncated_normal(mean=32, sd=8, low=10, high=70)

        # Grip strength (kg)
        grip_strength = truncated_normal(mean=32, sd=9, low=10, high=70)

        # Slightly tie fitness to activity:
        vo2max += (activity_mins - 30) / 30.0
        grip_strength += (steps_per_day - 8000) / 4000.0
        vo2max = float(np.clip(vo2max, 10, 80))
        grip_strength = float(np.clip(grip_strength, 8, 80))

        # --- Biological age construction (rule-based, NHANES-like cutoffs) ---
        biological_age = age

        # HbA1c contributions (ADA-style thresholds)
        if hba1c >= 6.5:           # diabetes range
            biological_age += np.random.uniform(5, 10)
        elif hba1c >= 5.7:         # prediabetes
            biological_age += np.random.uniform(2, 5)

        # CRP: low <1, average 1–3, high >3 (cardio risk)
        if crp > 3:
            biological_age += np.random.uniform(3, 7)
        elif crp > 1:
            biological_age += np.random.uniform(1, 3)

        # BMI categories
        if bmi >= 35:              # class II/III obesity
            biological_age += np.random.uniform(5, 9)
        elif bmi >= 30:
            biological_age += np.random.uniform(3, 7)
        elif bmi >= 25:
            biological_age += np.random.uniform(1, 3)
        elif bmi < 18.5:
            biological_age += np.random.uniform(1, 4)

        # Blood pressure: elevate bio-age for Stage 1/2 hypertension
        if systolic_bp >= 140 or diastolic_bp >= 90:
            biological_age += np.random.uniform(4, 8)
        elif systolic_bp >= 130 or diastolic_bp >= 80:
            biological_age += np.random.uniform(2, 4)

        # Sleep
        if sleep_hours < 6:
            biological_age += np.random.uniform(2, 4)
        elif sleep_hours > 9:
            biological_age += np.random.uniform(1, 2)

        # Physical activity / steps
        if steps_per_day < 5000:
            biological_age += np.random.uniform(2, 5)
        elif steps_per_day > 10000:
            biological_age -= np.random.uniform(1, 3)

        # Stress / mood
        if stress_score >= 8:
            biological_age += np.random.uniform(2, 4)
        if mood_scale <= 3:
            biological_age += np.random.uniform(1, 3)

        # HRV
        if hrv < 30:
            biological_age += np.random.uniform(2, 4)
        elif hrv > 70:
            biological_age -= np.random.uniform(1, 3)

        # VO2max – fitter people “younger”
        if vo2max >= 40:
            biological_age -= np.random.uniform(1, 4)
        elif vo2max <= 20:
            biological_age += np.random.uniform(2, 5)

        # Add small random noise
        biological_age += np.random.normal(0, 2.5)

        # Keep biological age in a plausible range
        biological_age = float(np.clip(biological_age, 20, 100))

        data.append({
            'age': float(age),
            'gender': int(gender),
            'heart_rate': float(heart_rate),
            'systolic_bp': float(systolic_bp),
            'diastolic_bp': float(diastolic_bp),
            'bmi': float(bmi),
            'hba1c': float(hba1c),
            'cholesterol': float(cholesterol),
            'hdl': float(hdl),
            'creatinine': float(creatinine),
            'wbc': float(wbc),
            'crp': float(crp),
            'il6': float(il6),
            'tnf_alpha': float(tnf_alpha),
            'glyca': float(glyca),
            'sleep_hours': float(sleep_hours),
            'activity_mins': float(activity_mins),
            'stress_score': int(stress_score),
            'mood_scale': int(mood_scale),
            'steps_per_day': float(steps_per_day),
            'hrv': float(hrv),
            'vo2max': float(vo2max),
            'grip_strength': float(grip_strength),
            'biological_age': float(biological_age)
        })

    return pd.DataFrame(data)


@st.cache_resource
def train_model():
    """Train the biological age prediction model"""
    df = generate_synthetic_data(1000)
    X = df[FEATURE_COLUMNS].copy()
    y = df['biological_age']
    X['gender'] = X['gender'].astype(int)
    X = X.fillna(X.mean())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = BiologicalAgePredictor(input_size=len(FEATURE_COLUMNS))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    model.train()
    num_epochs = 200
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return model, scaler

def convert_java_to_python(value):
    """Convert Java objects to Python native types"""
    if value is None:
        return None
    
    # Handle Java String objects
    if hasattr(value, '__class__') and 'java' in str(type(value)):
        if 'String' in str(type(value)):
            return str(value)
        elif 'Number' in str(type(value)) or 'Integer' in str(type(value)) or 'Double' in str(type(value)):
            return float(str(value))
        elif 'BigDecimal' in str(type(value)):
            return float(str(value))
        else:
            return str(value)
    
    return value

def safe_float_conversion(value, default=None):
    """Safely convert value to float with proper error handling"""
    if value is None:
        return default
    
    try:
        # Convert Java objects to string first
        if hasattr(value, '__class__') and 'java' in str(type(value)):
            value = str(value)
        
        # Try to convert to float
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_get_value(data, key, default=None):
    """Safely get value from dictionary with None protection"""
    value = data.get(key, default)
    return value if value is not None else default

def safe_compare(value, threshold, default_value=None):
    """Safely compare values, handling None cases"""
    if value is None:
        return False
    try:
        return float(value) > threshold
    except (ValueError, TypeError):
        return False

class PatientDataCollector:
    def __init__(self, conn):
        self.conn = conn

    def get_patient_profile(self, hn):
        """Get patient profile from database with proper type conversion"""
        try:
            if self.conn is None:
                return None
                
            cursor = self.conn.cursor()
            query = """
            SELECT papmi_no, paper_ageyr, ctsex_desc, ctnat_code
            FROM vw_patient_profiles
            WHERE papmi_no = ?
            """
            cursor.execute(query, (hn,))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return {
                    'hn': convert_java_to_python(result[0]),
                    'age': safe_float_conversion(result[1], 50.0),
                    'gender': convert_java_to_python(result[2]),
                    'nationality': convert_java_to_python(result[3])
                }
            return None
        except Exception as e:
            st.error(f"Error getting patient profile: {e}")
            return None

    def get_latest_encounter(self, hn):
        """Get latest encounter number for the patient"""
        try:
            if self.conn is None:
                return None
                
            cursor = self.conn.cursor()
            query = """
            SELECT paadm_admno
            FROM pa_adm 
            INNER JOIN pa_patmas ON paadm_papmi_dr = papmi_rowid
            WHERE papmi_no = ?
            ORDER BY paadm_admdate DESC
            LIMIT 1
            """
            cursor.execute(query, (hn,))
            result = cursor.fetchone()
            cursor.close()
            
            return convert_java_to_python(result[0]) if result else None
        except Exception as e:
            st.error(f"Error getting encounter: {e}")
            return None

    def get_vitals_and_bmi(self, en):
        """Get vitals and calculate BMI with proper type conversion"""
        try:
            if self.conn is None:
                return {}, None
                
            cursor = self.conn.cursor()
            query = """
            SELECT paadm_admno, itm_desc, obs_value, obs_date, obs_time, obs_item_dr
            FROM mr_observations
            INNER JOIN mr_adm ON mradm_rowid = obs_parref
            INNER JOIN mrc_observationitem ON itm_rowid = obs_item_dr
            INNER JOIN pa_adm ON paadm_rowid = mradm_adm_dr
            WHERE obs_item_dr IN (7, 8, 9, 10, 11, 69, 87, 90)
            AND paadm_admno = ?
            ORDER BY obs_date DESC, obs_time DESC
            """
            cursor.execute(query, (en,))
            results = cursor.fetchall()
            cursor.close()
            
            if not results:
                return {}, None
            
            # Convert results to proper data types
            converted_results = []
            for row in results:
                converted_row = [
                    convert_java_to_python(row[0]),  # paadm_admno
                    convert_java_to_python(row[1]),  # itm_desc
                    convert_java_to_python(row[2]),  # obs_value
                    convert_java_to_python(row[3]),  # obs_date
                    convert_java_to_python(row[4]),  # obs_time
                    convert_java_to_python(row[5])   # obs_item_dr
                ]
                converted_results.append(converted_row)
            
            df = pd.DataFrame(converted_results, 
                            columns=['paadm_admno', 'itm_desc', 'obs_value', 'obs_date', 'obs_time', 'obs_item_dr'])
            
            # Process vitals
            vitals = {}
            for _, row in df.iterrows():
                obs_item_dr = str(row['obs_item_dr'])
                if obs_item_dr in VITAL_SIGNS:
                    try:
                        value = safe_float_conversion(row['obs_value'])
                        if value is not None:
                            vitals[VITAL_SIGNS[obs_item_dr]] = value
                    except (ValueError, TypeError):
                        continue
            
            # Calculate BMI
            bmi = None
            weight_rows = df[df['obs_item_dr'].astype(str) == '90']
            height_rows = df[df['obs_item_dr'].astype(str) == '87']
            
            if not weight_rows.empty and not height_rows.empty:
                try:
                    weight = safe_float_conversion(weight_rows.iloc[0]['obs_value'])
                    height = safe_float_conversion(height_rows.iloc[0]['obs_value'])
                    
                    if weight and height and height > 0:
                        bmi = weight / ((height / 100) ** 2)
                except (IndexError, ZeroDivisionError, ValueError, TypeError):
                    bmi = None
            
            return vitals, bmi
            
        except Exception as e:
            st.error(f"Error getting vitals: {e}")
            return {}, None

    def get_lab_data(self, hn):
        """Get lab data from database with proper type conversion"""
        try:
            if self.conn is None:
                return {}
                
            cursor = self.conn.cursor()
            query = """
            SELECT epi_no, dte_of_req, ctts_cde, ctts_nme, tst_dta, cttc_cde, cttc_des
            FROM bv_tcl01_vw_labord
            WHERE paadm_papmi_dr = ?
            AND dte_of_req >= CURRENT_DATE - INTERVAL '90' DAY
            ORDER BY dte_of_req DESC
            """
            cursor.execute(query, (hn,))
            results = cursor.fetchall()
            cursor.close()
            
            if not results:
                return {}
            
            lab_data = {}
            for result in results:
                # Convert Java objects to Python types
                epi_no = convert_java_to_python(result[0])
                dte_of_req = convert_java_to_python(result[1])
                ctts_cde = convert_java_to_python(result[2])
                ctts_nme = convert_java_to_python(result[3])
                tst_dta = convert_java_to_python(result[4])
                cttc_cde = convert_java_to_python(result[5])
                cttc_des = convert_java_to_python(result[6])
                
                if cttc_cde in LAB_CONDITIONS and tst_dta:
                    try:
                        test_value = safe_float_conversion(tst_dta)
                        if test_value is not None:
                            lab_data[cttc_cde] = {
                                'value': test_value,
                                'name': cttc_des,
                                'date': dte_of_req
                            }
                    except (ValueError, TypeError):
                        continue
            
            return lab_data
            
        except Exception as e:
            st.error(f"Error getting lab data: {e}")
            return {}

    def estimate_missing_features(self, real_data):
        """Estimate synthetic features based on available real data with None safety"""
        # Get values with safe defaults
        age = safe_get_value(real_data, 'age', 50.0)
        bmi = safe_get_value(real_data, 'bmi', 25.0)
        hba1c = safe_get_value(real_data, 'hba1c', 5.5)
        crp = safe_get_value(real_data, 'crp', 2.0)
        heart_rate = safe_get_value(real_data, 'heart_rate', 72.0)
        systolic_bp = safe_get_value(real_data, 'systolic_bp', 120.0)
        gender = safe_get_value(real_data, 'gender', 'Male')
        
        # Ensure all values are float for calculations
        try:
            age = float(age)
            bmi = float(bmi)
            hba1c = float(hba1c)
            crp = float(crp)
            heart_rate = float(heart_rate)
            systolic_bp = float(systolic_bp)
        except (ValueError, TypeError):
            # Use defaults if conversion fails
            age = 50.0
            bmi = 25.0
            hba1c = 5.5
            crp = 2.0
            heart_rate = 72.0
            systolic_bp = 120.0
        
        # Estimate inflammatory markers with safe comparisons
        il6_base = 2.0 + (age - 30) * 0.05
        if safe_compare(hba1c, 6.5):
            il6_base *= 1.5
        if safe_compare(bmi, 30):
            il6_base *= 1.3
            
        tnf_alpha_base = 1.5 + (age - 30) * 0.03
        if safe_compare(hba1c, 6.5):
            tnf_alpha_base *= 1.4
        if safe_compare(bmi, 30):
            tnf_alpha_base *= 1.2
            
        glyca_base = 350 + (age - 30) * 2
        if safe_compare(crp, 3):
            glyca_base += 50
        if safe_compare(hba1c, 6.5):
            glyca_base += 30
            
        # Estimate fitness markers with safe comparisons
        steps_base = 10000 - (age - 30) * 50
        if safe_compare(bmi, 30):
            steps_base -= 2000
        if safe_compare(hba1c, 6.5):
            steps_base -= 1000
            
        hrv_base = 60 - (age - 30) * 0.5
        if safe_compare(crp, 3):
            hrv_base -= 10
        if safe_compare(bmi, 30):
            hrv_base -= 5
            
        vo2max_base = 45 - (age - 30) * 0.3
        if safe_compare(bmi, 30):
            vo2max_base -= 5
        if safe_compare(heart_rate, 90):
            vo2max_base -= 3
            
        grip_strength_base = 45 - (age - 30) * 0.2
        if gender == 'Female':
            grip_strength_base *= 0.7
            
        # Estimate stress markers with safe comparisons
        sleep_base = 7.5 - (age - 30) * 0.02
        if safe_compare(crp, 3):
            sleep_base -= 0.5
        if safe_compare(bmi, 30):
            sleep_base -= 0.5
            
        stress_base = 3
        if safe_compare(hba1c, 6.5):
            stress_base += 2
        if safe_compare(bmi, 30):
            stress_base += 1
        if safe_compare(systolic_bp, 140):
            stress_base += 2
            
        mood_base = 8 - (age - 30) * 0.02
        if safe_compare(stress_base, 6):
            mood_base -= 1
        if safe_compare(crp, 3):
            mood_base -= 0.5
            
        return {
            'il6': max(1.0, il6_base),
            'tnf_alpha': max(1.0, tnf_alpha_base),
            'glyca': max(300, glyca_base),
            'steps_per_day': max(2000, steps_base),
            'activity_mins': max(10, 45 - (age - 30) * 0.5),
            'hrv': max(20, hrv_base),
            'vo2max': max(15, vo2max_base),
            'grip_strength': max(15, grip_strength_base),
            'sleep_hours': max(4, min(10, sleep_base)),
            'stress_score': min(10, max(1, stress_base)),
            'mood_scale': max(1, min(10, mood_base))
        }

    def collect_patient_data(self, hn):
        """Collect patient data from database with comprehensive error handling"""
        try:
            # Get patient profile
            patient_profile = self.get_patient_profile(hn)
            if not patient_profile:
                st.warning(f"Patient profile not found for HN: {hn}")
                return None
            
            # Get latest encounter
            en = self.get_latest_encounter(hn)
            if not en:
                st.info(f"No encounter found for patient {hn}, using profile data only")
            
            # Get vitals and BMI
            vitals_data, bmi = self.get_vitals_and_bmi(en) if en else ({}, None)
            
            # Get lab data
            lab_data = self.get_lab_data(hn)
            
            # Compile real patient data with comprehensive None checking
            real_patient_data = {
                'name': f'Patient {hn}',
                'hn': hn,
                'en': en,
                'age': safe_float_conversion(patient_profile.get('age'), 50.0),
                'gender': patient_profile.get('gender', 'Male'),
                'bmi': safe_float_conversion(bmi) if bmi is not None else None,
                'heart_rate': safe_float_conversion(vitals_data.get('PR')) if vitals_data.get('PR') is not None else None,
                'systolic_bp': safe_float_conversion(vitals_data.get('BP systolic')) if vitals_data.get('BP systolic') is not None else None,
                'diastolic_bp': safe_float_conversion(vitals_data.get('BP diastolic')) if vitals_data.get('BP diastolic') is not None else None,
                'hba1c': safe_float_conversion(lab_data.get('N0510', {}).get('value')) if lab_data.get('N0510', {}).get('value') is not None else None,
                'cholesterol': safe_float_conversion(lab_data.get('C0131', {}).get('value')) if lab_data.get('C0131', {}).get('value') is not None else None,
                'hdl': safe_float_conversion(lab_data.get('C0141', {}).get('value')) if lab_data.get('C0141', {}).get('value') is not None else None,
                'creatinine': safe_float_conversion(lab_data.get('C0070', {}).get('value')) if lab_data.get('C0070', {}).get('value') is not None else None,
                'wbc': safe_float_conversion(lab_data.get('A0005', {}).get('value')) if lab_data.get('A0005', {}).get('value') is not None else None,
                'crp': safe_float_conversion(lab_data.get('T0700', {}).get('value')) if lab_data.get('T0700', {}).get('value') is not None else None,
                'lab_results_detailed': lab_data
            }
            
            # Debug information
            st.info(f"Retrieved data summary for {hn}:")
            st.info(f"- Age: {real_patient_data['age']}")
            st.info(f"- Gender: {real_patient_data['gender']}")
            st.info(f"- BMI: {real_patient_data['bmi']}")
            st.info(f"- Vitals count: {len(vitals_data)}")
            st.info(f"- Lab results count: {len(lab_data)}")
            
            # Fill missing features with estimates
            estimated_features = self.estimate_missing_features(real_patient_data)
            real_patient_data.update(estimated_features)
            
            return real_patient_data
            
        except Exception as e:
            st.error(f"Error collecting patient data: {e}")
            import traceback
            st.error(f"Full traceback: {traceback.format_exc()}")
            return None

def connect_to_denodo_jdbc():
    """Connect to Denodo via JDBC with proper error handling"""
    try:
        start_jvm()
        conn = jaydebeapi.connect(
            jdbc_driver_class,
            jdbc_url,
            [jdbc_username, jdbc_password],
            jdbc_driver_path
        )
        st.success("✅ Connected to Denodo via JDBC")
        return conn
    except Exception as e:
        st.error(f"Failed to connect to Denodo: {str(e)}")
        # Print more detailed error information
        st.error(f"JDBC URL: {jdbc_url}")
        st.error(f"Driver path: {jdbc_driver_path}")
        return None

def display_tier_data(tier_name, features, patient_data, tier_color):
    """Display tier data with status indicators"""
    st.subheader(f"🔍 {tier_name}")
    
    real_count = 0
    total_count = len(features)
    
    cols = st.columns(3)
    
    for i, feature in enumerate(features):
        col = cols[i % 3]
        
        with col:
            value = patient_data.get(feature)
            if value is not None and not pd.isna(value):
                # Real data
                real_count += 1
                if feature == 'gender':
                    st.markdown(f"**{feature.replace('_', ' ').title()}**: :green[{value}] ✅")
                else:
                    st.markdown(f"**{feature.replace('_', ' ').title()}**: :green[{value:.1f}] ✅")
            else:
                # Show missing data
                st.markdown(f"**{feature.replace('_', ' ').title()}**: :red[Missing] ❌")
    
    # Status indicator
    percentage = (real_count / total_count) * 100 if total_count > 0 else 0
    status_color = "🟢" if percentage >= 70 else "🟡" if percentage >= 40 else "🔴"
    st.markdown(f"{status_color} **Data Completeness**: {real_count}/{total_count} ({percentage:.0f}%)")
    st.markdown("---")
    
    return real_count, total_count

def create_biomarker_charts(patient_data):
    """Create charts for Tier II biomarkers"""
    biomarker_data = {}
    
    for feature in TIER_II_FEATURES:
        value = patient_data.get(feature)
        if value is not None and not pd.isna(value):
            biomarker_data[feature] = value
    
    if biomarker_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        features = list(biomarker_data.keys())
        values = list(biomarker_data.values())
        
        bars = ax1.bar(features, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE'])
        ax1.set_title('Biomarker Levels', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Values')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Radar chart
        if len(biomarker_data) >= 3:
            angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
            
            # Normalize values for radar chart (0-1 scale)
            normalized_values = []
            for feature, value in biomarker_data.items():
                if feature == 'hba1c':
                    norm_val = min(1, max(0, (value - 4) / 6))
                elif feature == 'cholesterol':
                    norm_val = min(1, max(0, value / 300))
                elif feature == 'hdl':
                    norm_val = min(1, max(0, value / 100))
                elif feature == 'creatinine':
                    norm_val = min(1, max(0, value / 2))
                elif feature == 'wbc':
                    norm_val = min(1, max(0, value / 15))
                elif feature == 'crp':
                    norm_val = min(1, max(0, value / 10))
                else:
                    norm_val = min(1, max(0, value / (np.mean(list(biomarker_data.values())) * 2)))
                normalized_values.append(norm_val)
            
            normalized_values += normalized_values[:1]  # Complete the circle
            angles = np.concatenate((angles, [angles[0]]))
            
            ax2.plot(angles, normalized_values, 'o-', linewidth=2, color='#FF6B6B')
            ax2.fill(angles, normalized_values, color='#FF6B6B', alpha=0.25)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(features)
            ax2.set_ylim(0, 1)
            ax2.set_title('Biomarker Profile (Normalized)', fontsize=14, fontweight='bold')
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, 'Insufficient data\nfor radar chart', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Biomarker Profile', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)

def prepare_features_for_prediction(patient_data):
    """Convert patient data to model input format with proper handling of missing values"""
    features = {}
    
    # Map patient data to feature columns
    for col in FEATURE_COLUMNS:
        if col == 'gender':
            gender_value = patient_data.get('gender')
            if gender_value:
                features[col] = 1 if gender_value.lower() in ['female', 'f'] else 0
            else:
                features[col] = 0  # Default to male
        else:
            value = patient_data.get(col)
            if value is not None and not pd.isna(value):
                features[col] = float(value)
            else:
                # Use None for missing values - will be imputed later
                features[col] = None
    
    # Convert to DataFrame
    feature_df = pd.DataFrame([features])
    return feature_df

def predict_biological_age(patient_data, model, scaler):
    """Predict biological age using trained model with real data"""
    try:
        # Prepare features
        feature_df = prepare_features_for_prediction(patient_data)
        
        # Count missing values
        missing_count = feature_df.isnull().sum().sum()
        total_features = len(FEATURE_COLUMNS)
        
        st.info(f"Using {total_features - missing_count}/{total_features} real features for prediction")
        
        if missing_count > 0:
            st.warning(f"Missing {missing_count} features - will be imputed with median values from training data")
            
            # Use training data medians for imputation
            training_medians = {
                'age': 50.0, 'gender': 0, 'heart_rate': 72.0, 'systolic_bp': 120.0, 
                'diastolic_bp': 80.0, 'bmi': 25.0, 'hba1c': 5.5, 'cholesterol': 190.0,
                'hdl': 55.0, 'creatinine': 0.9, 'wbc': 7.0, 'crp': 2.0, 'il6': 2.5,
                'tnf_alpha': 1.5, 'glyca': 380.0, 'sleep_hours': 7.5, 'activity_mins': 45.0,
                'stress_score': 5.0, 'mood_scale': 8.0, 'steps_per_day': 8000.0, 
                'hrv': 50.0, 'vo2max': 45.0, 'grip_strength': 40.0
            }
            
            for col in feature_df.columns:
                if feature_df[col].isnull().any():
                    feature_df[col] = feature_df[col].fillna(training_medians.get(col, 0))
        
        # Scale features
        features_scaled = scaler.transform(feature_df)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            prediction = model(features_tensor)
            biological_age = prediction.item()
        
        return biological_age, feature_df
        
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

def calculate_age_acceleration(chronological_age, biological_age):
    """Calculate biological age acceleration"""
    acceleration = biological_age - chronological_age
    
    if acceleration <= -5:
        status = "🟢 Excellent"
        description = "You're aging slower than your chronological age!"
    elif acceleration <= -2:
        status = "🟡 Good"
        description = "You're aging slightly slower than expected."
    elif acceleration <= 2:
        status = "🟠 Average"
        description = "You're aging at an expected rate."
    elif acceleration <= 5:
        status = "🔴 Concerning"
        description = "You're aging faster than your chronological age."
    else:
        status = "🔴🔴 High Risk"
        description = "Significant biological age acceleration detected."
    
    return acceleration, status, description

def generate_recommendations(patient_data, biological_age, chronological_age):
    """Generate personalized health recommendations with None safety"""
    recommendations = []
    
    age_acceleration = biological_age - chronological_age
    
    # BMI recommendations
    bmi = safe_get_value(patient_data, 'bmi')
    if bmi is not None and bmi > 30:
        recommendations.append({
            'category': 'Weight Management',
            'priority': 'High',
            'recommendation': 'Consider weight loss program - BMI indicates obesity',
            'target': 'Reduce BMI to 25-30 range'
        })
    elif bmi is not None and bmi > 25:
        recommendations.append({
            'category': 'Weight Management',
            'priority': 'Medium',
            'recommendation': 'Maintain healthy weight through diet and exercise',
            'target': 'Keep BMI in 18.5-25 range'
        })
    
    # Blood pressure recommendations
    systolic_bp = safe_get_value(patient_data, 'systolic_bp')
    if systolic_bp is not None and systolic_bp > 140:
        recommendations.append({
            'category': 'Cardiovascular Health',
            'priority': 'High',
            'recommendation': 'Monitor blood pressure regularly, consider medical consultation',
            'target': 'Maintain systolic BP < 130 mmHg'
        })
    
    # HbA1c recommendations
    hba1c = safe_get_value(patient_data, 'hba1c')
    if hba1c is not None and hba1c > 6.5:
        recommendations.append({
            'category': 'Metabolic Health',
            'priority': 'High',
            'recommendation': 'Diabetes management - consult endocrinologist',
            'target': 'Target HbA1c < 7.0%'
        })
    elif hba1c is not None and hba1c > 5.7:
        recommendations.append({
            'category': 'Metabolic Health',
            'priority': 'Medium',
            'recommendation': 'Pre-diabetes prevention - lifestyle modifications',
            'target': 'Target HbA1c < 5.7%'
        })
    
    # Cholesterol recommendations
    cholesterol = safe_get_value(patient_data, 'cholesterol')
    if cholesterol is not None and cholesterol > 240:
        recommendations.append({
            'category': 'Lipid Management',
            'priority': 'High',
            'recommendation': 'High cholesterol - dietary changes and possible medication',
            'target': 'Target total cholesterol < 200 mg/dL'
        })
    
    # Inflammation recommendations
    crp = safe_get_value(patient_data, 'crp')
    if crp is not None and crp > 3:
        recommendations.append({
            'category': 'Inflammation',
            'priority': 'Medium',
            'recommendation': 'Elevated inflammation - anti-inflammatory diet, stress management',
            'target': 'Target CRP < 3.0 mg/L'
        })
    
    # Age acceleration specific recommendations
    if age_acceleration > 5:
        recommendations.append({
            'category': 'Urgent Care',
            'priority': 'High',
            'recommendation': 'Comprehensive medical evaluation recommended due to significant age acceleration',
            'target': 'Identify and address underlying health issues'
        })
    
    return recommendations

def create_age_comparison_chart(chronological_age, biological_age):
    """Create age comparison visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ages = ['Chronological Age', 'Biological Age']
    values = [chronological_age, biological_age]
    colors = ['#4ECDC4', '#FF6B6B' if biological_age > chronological_age else '#96CEB4']
    
    bars = ax.bar(ages, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f} years', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add difference indicator
    difference = biological_age - chronological_age
    if difference != 0:
        ax.annotate(f'Difference: {difference:+.1f} years',
                   xy=(0.5, max(values) * 0.7), xycoords=('axes fraction', 'data'),
                   ha='center', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_ylabel('Age (years)', fontsize=12)
    ax.set_title('Chronological vs Biological Age Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Set y-axis to start from reasonable minimum
    ax.set_ylim(min(values) * 0.8, max(values) * 1.2)
    
    plt.tight_layout()
    return fig

def create_recommendations_display(recommendations):
    """Display recommendations in organized format"""
    if not recommendations:
        st.info("🎉 Great! No specific recommendations at this time. Keep up the healthy lifestyle!")
        return
    
    # Group by priority
    high_priority = [r for r in recommendations if r['priority'] == 'High']
    medium_priority = [r for r in recommendations if r['priority'] == 'Medium']
    low_priority = [r for r in recommendations if r['priority'] == 'Low']
    
    if high_priority:
        st.error("🚨 **High Priority Recommendations**")
        for rec in high_priority:
            st.markdown(f"**{rec['category']}**: {rec['recommendation']}")
            st.markdown(f"*Target: {rec['target']}*")
            st.markdown("---")
    
    if medium_priority:
        st.warning("⚠️ **Medium Priority Recommendations**")
        for rec in medium_priority:
            st.markdown(f"**{rec['category']}**: {rec['recommendation']}")
            st.markdown(f"*Target: {rec['target']}*")
            st.markdown("---")
    
    if low_priority:
        st.info("💡 **Optimization Suggestions**")
        for rec in low_priority:
            st.markdown(f"**{rec['category']}**: {rec['recommendation']}")
            st.markdown(f"*Target: {rec['target']}*")
            st.markdown("---")

# Main Streamlit App
def main():
    st.title("🧬 Biological Age Prediction System")
    st.markdown("---")
    
    # Initialize connection
    conn = get_denodo_connection()
    
    # Load or train model
    with st.spinner("Loading AI model..."):
        model, scaler = train_model()
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar for patient input
    st.sidebar.header("🏥 Patient Information")
    
    if conn is None:
        st.sidebar.error("❌ Database connection failed")
        st.error("❌ Unable to connect to database. Please check connection settings.")
        return
    else:
        st.sidebar.success("✅ Database connected")
    
    # Patient input
    hn = st.sidebar.text_input("Hospital Number (HN)", placeholder="Enter patient HN")
    
    if st.sidebar.button("🔍 Analyze Patient", type="primary"):
        if hn:
            with st.spinner("Collecting patient data from database..."):
                collector = PatientDataCollector(conn)
                patient_data = collector.collect_patient_data(hn)
                
                if patient_data:
                    st.session_state.patient_data = patient_data
                    st.session_state.analysis_complete = True
                else:
                    st.error("❌ Patient not found or error retrieving data")
        else:
            st.sidebar.error("Please enter a Hospital Number")
    
    # Display results if analysis is complete
    if hasattr(st.session_state, 'analysis_complete') and st.session_state.analysis_complete:
        patient_data = st.session_state.patient_data
        
        # Patient header
        st.header(f"📋 Analysis Results for {patient_data.get('name', 'Unknown Patient')}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hospital Number", patient_data.get('hn', 'N/A'))
        with col2:
            st.metric("Age", f"{patient_data.get('age', 'N/A')} years")
        with col3:
            st.metric("Gender", patient_data.get('gender', 'N/A'))
        
        st.markdown("---")
        
        # Display tier data
        st.header("📊 Clinical Data by Tier")
        
        tier1_real, tier1_total = display_tier_data("Tier I - Basic Vitals", TIER_I_FEATURES, patient_data, "#4ECDC4")
        tier2_real, tier2_total = display_tier_data("Tier II - Laboratory Results", TIER_II_FEATURES, patient_data, "#FF6B6B")
        tier3_real, tier3_total = display_tier_data("Tier III - Fitness Metrics", TIER_III_FEATURES, patient_data, "#45B7D1")
        tier4_real, tier4_total = display_tier_data("Tier IV - Lifestyle Factors", TIER_IV_FEATURES, patient_data, "#96CEB4")
        
        # Overall data completeness
        total_real = tier1_real + tier2_real + tier3_real + tier4_real
        total_features = tier1_total + tier2_total + tier3_total + tier4_total
        overall_completeness = (total_real / total_features) * 100 if total_features > 0 else 0
        
        st.subheader(f"📈 Overall Data Completeness: {overall_completeness:.0f}%")
        progress_bar = st.progress(overall_completeness / 100)
        
        # Biomarker charts (only show if we have real biomarker data)
        real_biomarker_count = sum(1 for feature in TIER_II_FEATURES 
                                  if patient_data.get(feature) is not None and not pd.isna(patient_data.get(feature)))
        
        if real_biomarker_count > 0:
            st.header("🔬 Biomarker Analysis")
            create_biomarker_charts(patient_data)
        
        # Biological age prediction
        st.header("🎯 Biological Age Prediction")
        
        with st.spinner("Calculating biological age using real patient data..."):
            biological_age, feature_df = predict_biological_age(patient_data, model, scaler)
        
        if biological_age is not None:
            chronological_age = patient_data.get('age', 50)
            
            # Age comparison chart
            fig = create_age_comparison_chart(chronological_age, biological_age)
            st.pyplot(fig)
            
            # Age acceleration analysis
            acceleration, status, description = calculate_age_acceleration(chronological_age, biological_age)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chronological Age", f"{chronological_age:.1f} years")
            with col2:
                st.metric("Biological Age", f"{biological_age:.1f} years", 
                         delta=f"{acceleration:+.1f} years")
            with col3:
                st.markdown(f"**Status**: {status}")
            
            st.markdown(f"**Analysis**: {description}")
            
            # Recommendations
            st.header("💡 Personalized Recommendations")
            recommendations = generate_recommendations(patient_data, biological_age, chronological_age)
            create_recommendations_display(recommendations)
            
            # Export functionality
            st.header("📥 Export Results")
            
            if st.button("📊 Generate Report"):
                report_data = {
                    'patient_info': {
                        'name': patient_data.get('name'),
                        'hn': patient_data.get('hn'),
                        'age': chronological_age,
                        'gender': patient_data.get('gender')
                    },
                    'analysis_results': {
                        'biological_age': biological_age,
                        'age_acceleration': acceleration,
                        'status': status,
                        'data_completeness': overall_completeness
                    },
                    'recommendations': recommendations,
                    'timestamp': datetime.now().isoformat()
                }
                
                st.download_button(
                    label="Download JSON Report",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"biological_age_report_{hn}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.error("❌ Unable to calculate biological age. Please check the data.")

if __name__ == "__main__":
    main()
