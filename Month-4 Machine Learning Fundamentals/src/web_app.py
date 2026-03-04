from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.abspath(".."))

from src.model_inference import predict_churn

app = FastAPI(title="Customer Churn Prediction API")


class CustomerInput(BaseModel):
    Gender: str
    Senior_Citizen: int
    Partner: str
    Dependents: str
    Tenure_Months: int
    Phone_Service: str
    Multiple_Lines: str
    Internet_Service: str
    Online_Security: str
    Online_Backup: str
    Device_Protection: str
    Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str
    Monthly_Charges: float
    Total_Charges: float
    CLTV: float


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: CustomerInput):
    input_dict = {
        "Gender": customer.Gender,
        "Senior Citizen": customer.Senior_Citizen,
        "Partner": customer.Partner,
        "Dependents": customer.Dependents,
        "Tenure Months": customer.Tenure_Months,
        "Phone Service": customer.Phone_Service,
        "Multiple Lines": customer.Multiple_Lines,
        "Internet Service": customer.Internet_Service,
        "Online Security": customer.Online_Security,
        "Online Backup": customer.Online_Backup,
        "Device Protection": customer.Device_Protection,
        "Tech Support": customer.Tech_Support,
        "Streaming TV": customer.Streaming_TV,
        "Streaming Movies": customer.Streaming_Movies,
        "Contract": customer.Contract,
        "Paperless Billing": customer.Paperless_Billing,
        "Payment Method": customer.Payment_Method,
        "Monthly Charges": customer.Monthly_Charges,
        "Total Charges": customer.Total_Charges,
        "CLTV": customer.CLTV,
    }

    result = predict_churn(input_dict)

    return result