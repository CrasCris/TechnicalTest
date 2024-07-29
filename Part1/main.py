from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from joblib import dump, load
import pandas as pd

app = FastAPI(title='Churn API Predict')
# Load the best classifier and onehotencoder
best_classifier_loaded = load('best_classifier.joblib')
onehotencoder_geo = load('onehotencoder.pkl')

from typing import Dict
from pydantic import BaseModel

class request(BaseModel):
    '''
    CreditScore:int\n
    Geography:str | "Germany" , "France" , "Spain"\n
    Gender:str | "Female", "Male"\n
    Age:int\n
    Tenure:int\n
    Balance:float\n
    NumOfProducts:int\n
    HasCrCard:int\n
    IsActiveMember:int\n
    EstimatedSalary:float
    '''
    CreditScore:int
    Geography:str
    Gender:str
    Age:int
    Tenure:int
    Balance:float
    NumOfProducts:int
    HasCrCard:int
    IsActiveMember:int
    EstimatedSalary:float
    model_config = {
        "json_schema_extra": {
            "examples": [
                            {
                            "CreditScore": 59226,
                            "Geography": "Germany",
                            "Gender": 'Female',
                            "Age": 15,
                            "Tenure": 4,
                            "Balance": 51651.66,
                            "NumOfProducts": 3,
                            "HasCrCard": 0,
                            "IsActiveMember": 0,
                            "EstimatedSalary": 591265.615
                            }
            ]
        }
    }

class Customer:
    '''
        All the information of a customer and the data transform to use the model we have trained
    '''

    def __init__(self, CreditScore, Geography, Gender, Age, Tenure, Balance, 
                 NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
        self.CreditScore = CreditScore
        self.Geography = Geography
        self.Gender = Gender
        self.Age = Age
        self.Tenure = Tenure
        self.Balance = Balance
        self.NumOfProducts = NumOfProducts
        self.HasCrCard = HasCrCard
        self.IsActiveMember = IsActiveMember
        self.EstimatedSalary = EstimatedSalary

    def __repr__(self):
        return (f"Customer(CreditScore={self.CreditScore}, Geography='{self.Geography}', "
                f"Gender='{self.Gender}', Age={self.Age}, Tenure={self.Tenure}, "
                f"Balance={self.Balance}, NumOfProducts={self.NumOfProducts}, "
                f"HasCrCard={self.HasCrCard}, IsActiveMember={self.IsActiveMember}, "
                f"EstimatedSalary={self.EstimatedSalary})")

    def transform_features(self):
        ''' Create a DataFrame with a single row (this customer)'''
        df = pd.DataFrame([{
            'CreditScore': self.CreditScore,
            'Geography': self.Geography,
            'Gender': self.Gender,
            'Age': self.Age,
            'Tenure': self.Tenure,
            'Balance': self.Balance,
            'NumOfProducts': self.NumOfProducts,
            'HasCrCard': self.HasCrCard,
            'IsActiveMember': self.IsActiveMember,
            'EstimatedSalary': self.EstimatedSalary
        }])
        
        # Transform categorical values
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
        df_geo_encoded = onehotencoder_geo.transform(df[['Geography']])
        df_geo_encoded = pd.DataFrame(df_geo_encoded, columns=onehotencoder_geo.get_feature_names_out(['Geography']))
        df = df.drop(columns=['Geography'])
        df = pd.concat([df_geo_encoded, df], axis=1)
        return df




@app.get("/")
async def read_root():
    context = """
    <html>
	<head>
		<title>Churn API Predict</title>
	</head>
    <body>Simple backend api to predict the possible clint Churn</body>
    </html>
    """
    return HTMLResponse(context)


@app.post('/predict_Churn')
async def predict(request:request):
    '''
        Prediction of a possible Churn customer
    '''

    cliente = Customer(request.CreditScore,request.Geography,request.Gender,
                       request.Age,request.Tenure,request.Balance,
                       request.NumOfProducts,request.HasCrCard,
                       request.IsActiveMember,request.EstimatedSalary)
    df_transformed = cliente.transform_features()
    X = df_transformed.iloc[:].values
    Y = best_classifier_loaded.predict(X)
    Y = Y.tolist()
    return_dict = {"Churn":Y[0]}
    return (return_dict)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8080, reload=True)