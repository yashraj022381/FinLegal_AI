import gradio as gr
import pandas as pd
import joblib

try:
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')

except FileNotFoundError as e:
    print(f"Missing file: {e}")
    exit()

scaler_expected_columns = [
    'Amount',
    'Is_International',
    'Is_Chip',
    'Is_Pin_Used',
    'Distance_From_Home',
    'Hour_of_Day',
    'Day_of_Week',
    'Is_Weekend',
    'Month'
    ]

def predict_fraud(amount, threshold, hour, distance, international, pin, chip, merchant):
    data = pd.DataFrame(0.0, index=[0], columns=feature_columns)

    # Fill columns (your existing code)
    data['Amount'] = amount
    data['Hour_of_Day'] = hour
    data['Distance_From_Home'] = distance
    data['Is_International'] = 1 if international else 0
    data['Is_Pin_Used'] = 1 if pin else 0
    data['Is_Chip'] = 1 if chip else 0

    if merchant != "None":
        col = f"Merchant_Category_{merchant}"
        if col in data.columns:
            data[col] = 1

    # Rule-based warning - make sure this runs
    rule_warning = ""
    suspicious_reasons = []
    
    if amount > 50000:
        suspicious_reasons.append("high amount")

    if hour < 6 or hour > 22:
        suspicious_reasons.append("strange time")

    if international:
        suspicious_reasons.append("international")

    if suspicious_reasons:
        rule_warning = "⚠ Suspicious pattern: " + " / ".join(suspicious_reasons) 

    else:
        rule_warning = "(no suspicious rules triggered)"
        
    try:
        # Scaling + prediction (your existing code)
        data_for_scaling = data[scaler_expected_columns]
        scaled = scaler.transform(data_for_scaling)
        data[scaler_expected_columns] = scaled

        pred_raw = model.predict(data)[0]
        anomaly_score = model.score_samples(data)[0]
        is_fraud = (pred_raw == -1) or (anomaly_score < threshold)

        # Better final result - prioritize rules if suspicious
        if is_fraud:
            final_result = "🚨 Fraud Alert! (ML detected)"
        elif suspicious_reasons:
            final_result = "⚠ Suspicious (rule-based warning)"
        else:
            final_result = "✅ Looks normal"

        score_text = f"Anomaly score: {anomaly_score:.4f} (lower = more suspicious)"

        # Return ALL three outputs
        return final_result, score_text, rule_warning

    except Exception as e:
        return "Error", f"Failed: {str(e)}", rule_warning

with gr.Blocks() as demo:
    gr.Markdown("# AI-Powered Fraud Detection")
    
    
    amount = gr.Number(label="Amount (₹)", value=500)
    threshold = gr.Slider(minimum=-1.0, maximum=0.0, value=-0.65, step=0.01, label="Anomaly Score Threshold (lower = more strict)",
                          info = "Adjust sensitivity: -0.5 catches more, -0.8 stricter")
    hour = gr.Slider(0, 23, value=14, label="Hour of Day")
    distance = gr.Number(label="Distance From Home (km)", value=5.0)
    international = gr.Checkbox(label="International Transaction?")
    pin = gr.Checkbox(label="PIN Used?", value=True)
    chip = gr.Checkbox(label="Chip Used?", value=True)
    merchant = gr.Dropdown(["None", "Electronics", "Food", "Fuel", "Groceries", "Online Services", "Travel", "Entertainment", "Health", "Utilities"],
                           label="Merchant Category")
    
    
    btn = gr.Button("Check for Fraud")
    output = [
        gr.Textbox(label="Result"),
        gr.Textbox(label="Anomaly Score"),
        gr.Textbox(label="Rule-based Warning")
    ]

    btn.click(
        predict_fraud,
        inputs=[amount, threshold, hour, distance, international, pin, chip, merchant],
        outputs=output
    )

demo.launch()
