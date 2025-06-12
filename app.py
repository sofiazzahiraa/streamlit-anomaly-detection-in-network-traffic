import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from modelSV import *

def load_model_and_scaler(model_path, scalerFeature_path, transformFeature_path, transformTarget_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scalerFeature_path)  
    le_F = joblib.load(transformFeature_path)
    le_T = joblib.load(transformTarget_path)
    return model, scaler, le_F, le_T

st.title("Prediksi dengan Model One-vs-One SVM")
st.write("Upload File Test (Format TXT atau CSV)")

uploaded_file = st.file_uploader("Pilih file CSV atau TXT", type=["csv", "txt"])

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]


if uploaded_file is not None:
    if uploaded_file.name.endswith('.txt'):
        delimiter = ","
    elif uploaded_file.name.endswith('.csv'):
        delimiter = ","
    

    df = pd.read_csv(uploaded_file, delimiter=delimiter,names=columns)

    st.write("Berikut adalah beberapa data dari file yang Anda unggah:")
    st.write(df.head())
    
    model, scaler, le_F, le_T = load_model_and_scaler('one_vs_one_svm_model.joblib', 'scaling.joblib', 'feature_encoders.joblib', 'target_encoder.joblib')

    target_column = 'attack_category'

    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    feature_encoders = {}

    for col in categorical_columns:
        df[col] = df[col].apply(lambda x: le_F[col].transform([x])[0] if x in le_F[col].classes_ else -1)  # -1 atau label default
       
    X_pred = df.drop(["label", "num_outbound_cmds"], axis=1) 

    X_pred_scaled = scaler.transform(X_pred)
    y_pred = model.predict(X_pred_scaled)
    
    y_pred_original = le_T.inverse_transform(y_pred)
    
    y_pred_df = pd.DataFrame(y_pred_original, columns=['predicted_attack_category'])

    st.write('Hasil Prediksi')
    st.write(y_pred_df)


    if len(df) > 10:
        st.write('Distribusi Hasil Prediksi')
        y_pred_counts = y_pred_df.value_counts()
        plt.figure(figsize=(8, 6))
        y_pred_counts.plot(kind='bar')
        plt.title('Distribusi Hasil Prediksi')
        plt.xlabel('Kelas')
        plt.ylabel('Frekuensi')
        plt.xticks(rotation=0)  
        st.pyplot(plt)
    else:
        pass
