import pandas as pd  
from sklearn.ensemble import IsolationForest  
import pickle  
import os  
  
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'activity_model.pkl')  
  
class IsolationForestModel:  
   def __init__(self):  
      # Inicializar o modelo de Isolation Forest  
      self.model = IsolationForest(contamination=0.1, random_state=42)  
  
   def train(self, data: pd.DataFrame):  
      # Treinar o modelo com os dados de atividade  
      self.model.fit(data)  
      # Salvar o modelo treinado  
      with open(MODEL_PATH, 'wb') as model_file:  
        pickle.dump(self.model, model_file)  
  
   def load_model(self):  
      # Check if the pickle file exists and is not empty  
      if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:  
        # Carregar o modelo treinado  
        with open(MODEL_PATH, 'rb') as model_file:  
           self.model = pickle.load(model_file)  
      else:  
        print("Pickle file is empty or does not exist. Re-training the model...")  
        self.train(pd.DataFrame({'timestamp': range(1, 101), 'events': [i + (i % 2) for i in range(1, 101)]}))  
  
   def predict(self, data: pd.DataFrame):  
      # Retornar previsões (1 = normal, -1 = anomalia)  
      return self.model.predict(data)  
  
# Função de treinamento para exemplo  
def train_model_example():  
   # Criar um exemplo de dataset de atividades (por exemplo, timestamp e número de eventos)  
   data = pd.DataFrame({  
      'timestamp': range(1, 101),  
      'events': [i + (i % 2) for i in range(1, 101)]  
   })  
   model = IsolationForestModel()  
   model.train(data)  
  
if __name__ == "__main__":  
   train_model_example()