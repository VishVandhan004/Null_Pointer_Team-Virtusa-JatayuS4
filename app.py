from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
from bson.objectid import ObjectId
import pickle
import joblib
import pandas as pd
import os
from llm import get_loan_explanation, get_fd_explanation

app = Flask(__name__)

with open(r"C:\Users\guna5\OneDrive\Desktop\Null Pointer   (Virtusa Hackathon)\Models\home_loan_model.pkl", 'rb') as f:
    model_objects = pickle.load(f)

model_pipeline = joblib.load(r"C:\Users\guna5\OneDrive\Desktop\Null Pointer   (Virtusa Hackathon)\Models\fd_model.pkl")

client = MongoClient('mongodb://localhost:27017/')
db = client['loan_db']
customers = db['customers']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/home')
def home_redirect():
    customer = customers.find_one(sort=[('_id', -1)])
    if customer:
        return redirect(url_for('predict', customer_id=str(customer['_id'])))
    else:
        return redirect(url_for('add_customer'))

@app.route('/employee_login.html')
def employee_login():
    return render_template('employee_login.html')

@app.route('/add', methods=['GET', 'POST'])
def add_customer():
    if request.method == 'POST':
        data = {
            'name': request.form['name'],
            'gender': request.form['gender'],
            'marital_status': request.form['marital_status'],
            'employment': request.form['employment'],
            'existing_customer': request.form['existing_customer'],
            'salary_account': request.form['salary_account'],
            'repayment_type': request.form['repayment_type'],
            'base_rate_type': request.form['base_rate_type'],
            'base_rate': float(request.form['base_rate']),
            'dependents': int(request.form['dependents']),
            'active_loans': int(request.form['active_loans']),
            'missed_payments': int(request.form['missed_payments']),
            'ltv': float(request.form['ltv']),
            'dti': float(request.form['dti']),
            'co_applicant_present': request.form['co_applicant_present'],
            'co_applicant_income': float(request.form.get('co_applicant_income', 0)),
        }
        customers.insert_one(data)
        return redirect(url_for('search_customer'))
    
    return render_template('add_customer.html')


@app.route('/search', methods=['GET', 'POST'])
def search_customer():
    results = []
    if request.method == 'POST':
        name = request.form['name']
        results = list(customers.find({'name': {'$regex': name, '$options': 'i'}}))
    return render_template('search_customer.html', customers=results)

@app.route('/view', methods=['GET', 'POST'])
def view_customers():
    customers_list = list(db.customers.find())
    return render_template('manage.html', customers=customers_list)

@app.route('/update_customer/<customer_id>', methods=['GET', 'POST'])
def update_customer(customer_id):
    customer = customers.find_one({'_id': ObjectId(customer_id)})
    if not customer:
        return "Customer not found", 404

    if request.method == 'POST':
        updated_data = {
            'name': request.form['name'],
            'gender': request.form['gender'],
            'marital_status': request.form['marital_status'],
            'employment': request.form['employment'],
            'existing_customer': request.form['existing_customer'],
            'salary_account': request.form['salary_account'],
            'repayment_type': request.form['repayment_type'],
            'base_rate_type': request.form['base_rate_type'],
            'base_rate': float(request.form['base_rate']),
            'dependents': int(request.form['dependents']),
            'active_loans': int(request.form['active_loans']),
            'missed_payments': int(request.form['missed_payments']),
            'ltv': float(request.form['ltv']),
            'dti': float(request.form['dti']),
            'co_applicant_present': request.form['co_applicant_present'],
            'co_applicant_income': float(request.form.get('co_applicant_income', 0)),
        }
        customers.update_one({'_id': ObjectId(customer_id)}, {'$set': updated_data})
        return redirect(url_for('search_customer'))

    return render_template('update_customer.html', customer=customer)


@app.route('/delete/<customer_id>', methods=['POST'])
def delete_customer(customer_id):
    try:
        result = customers.delete_one({'_id': ObjectId(customer_id)})
        if result.deleted_count == 0:
            return "Customer not found", 404
        return redirect(url_for('view_customers'))
    except Exception as e:
        return f"Error deleting customer: {e}", 400

@app.route('/predict/<customer_id>', methods=['GET', 'POST'])
def predict(customer_id):
    try:
        customer = customers.find_one({'_id': ObjectId(customer_id)})
    except Exception as e:
        return f"Invalid customer ID: {e}", 400

    if not customer:
        return "Customer not found", 404

    if request.method == 'POST':
        input_data = {
            'Age': int(request.form.get('age', 0)),  
            'Gender': customer.get('gender', ''),  
            'Marital Status': customer.get('marital_status', ''),
            'Dependents': customer.get('dependents', 0),  
            'Employment': customer.get('employment', ''),
            'Income (\u20b9)': int(request.form.get('income', 0)),  
            'Years in Job': int(request.form.get('years_in_job', 0)),  
            'Credit Score': int(request.form.get('credit_score', 0)),  
            'Active Loans': customer.get('active_loans', 0),  
            'Missed Payments (12M)': customer.get('missed_payments', 0),  
            'Loan Amount (\u20b9)': int(request.form.get('loan_amount', 0)), 
            'LTV (%)': customer.get('ltv', 0),  
            'DTI (%)': customer.get('dti', 0),  
            'Loan Tenure (yrs)': int(request.form.get('loan_tenure', 0)),  
            'Repayment Type': customer.get('repayment_type', ''),
            'Co-applicant Present': customer.get('co_applicant', ''),
            'Co-applicant Income (\u20b9)': customer.get('co_income', 0), 
            'Existing Customer': customer.get('existing_customer', ''),
            'Salary Account': customer.get('salary_account', ''),
            'Base Rate Type': customer.get('base_rate_type', '')
        }

        model = model_objects['model']
        scaler = model_objects['scaler']
        le_dict = model_objects['label_encoders']
        cat_cols = model_objects['categorical_columns']
        feature_columns = model_objects['feature_columns']

        input_df = pd.DataFrame([input_data])[feature_columns]
        
        for col in cat_cols:
            le = le_dict[col]
            input_df[col] = input_df[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
            input_df[col] = le.transform(input_df[col])

        input_df = input_df.apply(pd.to_numeric)
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)[0]

        explanation = get_loan_explanation(prediction, input_data)
        explanation_lines = [line.strip("-• ") for line in explanation.strip().splitlines() if line.strip()]

        return render_template(
            'predict_loan.html',
            customer=customer,
            customer_id=str(customer['_id']),
            prediction=f"{prediction:.2f}",
            explanation_lines=explanation_lines
        )

    return render_template('predict_loan.html', customer=customer, customer_id=str(customer['_id']))


@app.route('/predict-fd/<customer_id>', methods=['GET', 'POST'])
def predict_fd(customer_id):
    try:
        customer = customers.find_one({'_id': ObjectId(customer_id)})
    except Exception as e:
        return f"Invalid customer ID: {e}", 400

    if not customer:
        return "Customer not found", 404

    if request.method == 'POST':
        input_data = {
            'Age': int(request.form['age']),
            'Gender': customer.get('gender', ''),
            'Marital Status': customer.get('marital_status', ''),
            'Employment': customer.get('employment', ''),
            'Income (₹)': int(request.form['income']),
            'FD Amount (₹)': int(request.form['fd_amount']),
            'FD Tenure (yrs)': int(request.form['fd_tenure']),
            'Credit Score': int(request.form['credit_score']),
            'Base Rate (%)': customer.get('base_rate', 0),  
            'Base Rate Type': customer.get('base_rate_type', '')
        }

        input_df = pd.DataFrame([input_data])
        input_transformed = model_pipeline.named_steps['preprocessor'].transform(input_df)
        prediction = model_pipeline.named_steps['model'].predict(input_transformed)[0]

        final_rate = input_data['Base Rate (%)'] + prediction

        explanation = get_fd_explanation(prediction, input_data)
        explanation_lines = [line.strip("-• ") for line in explanation.strip().splitlines() if line.strip()]

        return render_template(
            'predict_fd.html',
            customer=customer,
            customer_id=str(customer['_id']),
            final_rate=f"{final_rate:.2f}",
            bps_adjustment=f"{prediction:.2f}",
            explanation_lines=explanation_lines
        )

    return render_template('predict_fd.html', customer=customer, customer_id=str(customer['_id']))


if __name__ == '__main__':
    app.run(debug=True)
