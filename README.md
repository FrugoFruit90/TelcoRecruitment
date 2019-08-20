## Console output

/home/janek/Documents/telco_churn/venv/bin/python /home/janek/Documents/telco_churn/main.py
gender ['Female' 'Male']  
Partner ['Yes' 'No']  
Dependents ['No' 'Yes']  
PhoneService ['No' 'Yes']  
MultipleLines ['No phone service' 'No' 'Yes']  
OnlineSecurity ['No' 'Yes' 'No internet service']  
OnlineBackup ['Yes' 'No' 'No internet service']  
DeviceProtection ['No' 'Yes' 'No internet service']  
TechSupport ['No' 'Yes' 'No internet service']  
StreamingTV ['No' 'Yes' 'No internet service']  
StreamingMovies ['No' 'Yes' 'No internet service']  
PaperlessBilling ['Yes' 'No']  
Churn ['No' 'Yes']  

LogisticRegression  
AUC: 0.813

(Bagged) SVM  
AUC: 0.777

RandomForestClassifier  
AUC: 0.814

ExtraTreesClassifier  
AUC: 0.815

GradientBoostingClassifier  
AUC: 0.814

Stacked Classifier  
AUC: 0.801

VotingClassifier  
AUC: 0.816

Process finished with exit code 0