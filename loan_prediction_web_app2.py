import numpy as np
import pickle
import streamlit as st 

# Load the trained model
loaded_model2 = pickle.load(open("C:/Users/mukul/Downloads/loan_model2/Loan_model2.sav", 'rb'))

def loan_prediction(input_data):
    # Convert input data to NumPy array and reshape for prediction
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)  # Ensure proper data type
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Make prediction
    prediction = loaded_model2.predict(input_data_reshaped)
    
    # Return appropriate response
    if prediction[0] == 0:
        return "Loan Not Approved"
    else:
        return "Loan Approved"

def main():
    st.title("Loan Approval Prediction")

    # Get user inputs
    Gender = st.text_input("Enter Gender (0 for Female, 1 for Male)")
    Married = st.text_input("Marital Status (0 for No, 1 for Yes)")
    Dependents = st.text_input("Number of Dependents")
    Education = st.text_input("Education (0 for Graduate, 1 for Not Graduate)")
    Self_Employed = st.text_input("Self Employed (0 for No, 1 for Yes)")
    ApplicantIncome = st.text_input("Applicant Income")
    CoapplicantIncome = st.text_input("Coapplicant Income")
    LoanAmount = st.text_input("Loan Amount")
    Loan_Amount_Term = st.text_input("Loan Amount Term")
    Credit_History = st.text_input("Credit History (0 for No, 1 for Yes)")

    loan_result = ""

    if st.button("Check Loan Approval"):
        # Convert inputs to numerical values
        try:
            input_data = [
                int(Gender), int(Married), int(Dependents), int(Education), int(Self_Employed),
                float(ApplicantIncome), float(CoapplicantIncome), float(LoanAmount),
                float(Loan_Amount_Term), int(Credit_History)
            ]
            loan_result = loan_prediction(input_data)
        except ValueError:
            loan_result = "Invalid input. Please enter valid numeric values."

    st.success(loan_result)

if __name__ == '__main__':
    main()
