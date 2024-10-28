from flask import Flask, jsonify  
from flask_cors import CORS  
from isolation_forest import IsolationForestModel  
  
app = Flask(__name__)  
CORS(app)  
  
# Carregar o modelo treinado  
model = IsolationForestModel()  
model.load_model()  

# Qualquer outra rota ou funcionalidade adicional do Flask pode ser mantida aqui.

if __name__ == "__main__":  
   app.run(debug=True, host="0.0.0.0", port=5000)
