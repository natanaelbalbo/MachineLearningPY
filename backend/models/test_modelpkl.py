import pickle  
  
with open('activity_model.pkl', 'rb') as model_file:  
  model = pickle.load(model_file, protocol=pickle.HIGHEST_PROTOCOL)  
  
print(model)
