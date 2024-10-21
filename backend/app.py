from flask import Flask, request, jsonify  
from flask_cors import CORS  
import pandas as pd  
import traceback  
from isolation_forest import IsolationForestModel  
from datetime import datetime  
  
app = Flask(__name__)  
CORS(app)  
  
# Carregar o modelo treinado  
model = IsolationForestModel()  
model.load_model()  
  
@app.route("/api/ia", methods=["POST"])  
def detect_anomalies():  
   try:  
      # Receber os dados JSON enviados pelo Flutter  
      data = request.get_json()  
      print(f"Dados recebidos: {data}")  
  
      # Verificar se os dados de senha estão presentes  
      password_data = data.get("password")  
      if password_data is None:  
        raise ValueError("Nenhuma senha fornecida")  
  
      # Preparar os dados para o modelo (simulando que a senha é parte dos dados de atividade)  
      df = pd.DataFrame({'timestamp': [int(datetime.now().timestamp())], 'events': [len(password_data)]})  
      print(f"Dados preparados para o modelo: {df}")  
    
      prediction = model.predict(df)[0]  # Retorna 1 (normal) ou -1 (anômalo)  
      print(f"Predição do modelo: {prediction}")  
  
      # Retornar a resposta para o Flutter  
      if prediction == 1:  
        return jsonify({"result": "Atividade normal"})  
      else:  
        return jsonify({"result": "Atividade suspeita"})  
  
   except Exception as e:  
      # Logar o erro completo com traceback  
      print(f"Erro: {str(e)}")  
      traceback.print_exc()  
      return jsonify({"error": str(e)}), 500  
  
if __name__ == "__main__":  
   app.run(debug=True, host="0.0.0.0", port=5000)