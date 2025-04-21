from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
from bson.objectid import ObjectId
import pickle
import joblib
import pandas as pd
import os

app = Flask(__name__)

with open(r"C:\Users\guna5\OneDrive\Desktop\HOME-LOAN\Models\home_loan_model.pkl", 'rb') as f:
    model_objects = pickle.load(f)

model_pipeline = joblib.load(r"C:\Users\guna5\OneDrive\Desktop\HOME-LOAN\Models\fd_model.pkl")

client = MongoClient('mongodb://localhost:27017/')
db = client['loan_db']
customers = db['customers']

@app.route('/')
def home():
    return redirect(url_for('add_customer'))

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
            'base_rate_type': request.form['base_rate_type']
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
            'base_rate_type': request.form['base_rate_type']
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
            'Age': int(request.form['age']),
            'Gender': customer['gender'],
            'Marital Status': customer['marital_status'],
            'Dependents': int(request.form['dependents']),
            'Employment': customer['employment'],
            'Income (\u20b9)': int(request.form['income']),
            'Years in Job': int(request.form['years_in_job']),
            'Credit Score': int(request.form['credit_score']),
            'Active Loans': int(request.form['active_loans']),
            'Missed Payments (12M)': int(request.form['missed_payments']),
            'Loan Amount (\u20b9)': int(request.form['loan_amount']),
            'LTV (%)': float(request.form['ltv']),
            'DTI (%)': float(request.form['dti']),
            'Loan Tenure (yrs)': int(request.form['loan_tenure']),
            'Repayment Type': customer['repayment_type'],
            'Co-applicant Present': request.form['co_applicant'],
            'Co-applicant Income (\u20b9)': int(request.form['co_income']),
            'Existing Customer': customer['existing_customer'],
            'Salary Account': customer['salary_account'],
            'Base Rate Type': customer['base_rate_type']
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

        return render_template(
            'predict_loan.html',
            customer=customer,
            customer_id=str(customer['_id']),
            prediction=f"{prediction:.2f}"
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
            'Gender': customer['gender'],
            'Marital Status': customer['marital_status'],
            'Employment': customer['employment'],
            'Income (₹)': int(request.form['income']),
            'FD Amount (₹)': int(request.form['fd_amount']),
            'FD Tenure (yrs)': int(request.form['fd_tenure']),
            'Credit Score': int(request.form['credit_score']),
            'Base Rate (%)': float(request.form['base_rate']),
            'Base Rate Type': customer['base_rate_type']
        }

        input_df = pd.DataFrame([input_data])
        prediction = model_pipeline.predict(input_df)[0]
        final_rate = input_data['Base Rate (%)'] + prediction

        return render_template(
            'predict_fd.html',
            customer=customer,
            customer_id=str(customer['_id']),
            final_rate=f"{final_rate:.2f}",
            bps_adjustment=f"{prediction:.2f}"
        )

    return render_template('predict_fd.html', customer=customer, customer_id=str(customer['_id']))

if __name__ == '__main__':
    app.run(debug=True)
