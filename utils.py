import pickle
import json
import config
import numpy as np

class MedicalInsurance():
    def __init__(self,age, sex, bmi, children,smoker, region):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = 'region_' + region

    def load_model(self):
        with open(config.MODEL_FILE_PATH, 'rb') as f:
            self.model = pickle.load(f)

        with open(config.JSON_FILE_PATH, 'r') as f:
            self.json_data = json.load(f) 

    def get_predicted_charges(self):
        self.load_model()
        test_array =  np.zeros(len(self.json_data['columns']))
        test_array[0] = self.age
        test_array[1] = self.json_data['sex'][self.sex]
        test_array[2] = self.bmi
        test_array[3] = self.children
        test_array[4] = self.json_data['smoker'][self.smoker]
        region_index = self.json_data['columns'].index(self.region)
        test_array[region_index] = 1

        print('Test Array >> ', test_array) 
        Predicted_charges = np.around(self.model.predict([test_array])[0],2)
        # print('Predicted Charges for Insurance are :RS.', Predicted_charges) 
        return Predicted_charges


if __name__ == "__main__":
    age = 41
    sex = 'male'
    bmi = 28
    children = 2
    smoker = 'no'
    region = 'southwest'
    med_ins = MedicalInsurance(age, sex, bmi, children,smoker, region)    
    med_ins.get_predicted_charges() 
