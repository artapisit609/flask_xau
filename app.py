from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import xgboost as xgb
import MetaTrader5 as mt5

app = Flask(__name__)

# โหลดโมเดล
lstm_model = tf.keras.models.load_model('lstm_model.keras')
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('xgb_model.json')

# ฟังก์ชันดึงข้อมูลจาก MT5
def fetch_data_from_mt5(symbol, timeframe, count):
    mt5.initialize()
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "H1": mt5.TIMEFRAME_H1,
        "D1": mt5.TIMEFRAME_D1
    }
    data = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe], 0, count)
    mt5.shutdown()
    return data

# หน้าเว็บหลัก
@app.route('/')
def index():
    return render_template('index.html')

# API สำหรับการทำนายผล
@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูลจากผู้ใช้
    user_input = request.json
    model_choice = user_input['model']
    timeframe = user_input['timeframe']
    parameters = user_input['parameters']
    input_data = np.array(user_input['data']).reshape(1, len(user_input['data']), -1)

    # ดึงข้อมูลจาก MT5 ตาม timeframe (ถ้าจำเป็น)
    symbol = "XAUUSDm"  # สามารถเปลี่ยนได้ตามต้องการ
    if timeframe:
        mt5_data = fetch_data_from_mt5(symbol, timeframe, count=100)
        # ใช้ข้อมูลจาก MT5 แทน input_data หรือประยุกต์ใช้งานตาม logic

    # เลือกโมเดลตามที่ผู้ใช้กำหนด
    if model_choice == "lstm":
        prediction = lstm_model.predict(input_data).flatten()
    elif model_choice == "xgboost":
        input_data_flat = input_data.reshape(1, -1)
        prediction = xgb_model.predict(input_data_flat)
    elif model_choice == "ensemble":
        lstm_pred = lstm_model.predict(input_data).flatten()
        input_data_flat = input_data.reshape(1, -1)
        xgb_pred = xgb_model.predict(input_data_flat)
        prediction = (lstm_pred + xgb_pred) / 2
    else:
        return jsonify({'error': 'Invalid model selection'}), 400

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(port=5005)
