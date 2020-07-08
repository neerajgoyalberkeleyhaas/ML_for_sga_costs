
# coding: utf-8

# In[3]:

#ERROR

#------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sqlite3

# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect("outsidein.db")
data = pd.read_sql_query("SELECT * from Training_database", con)
data.to_sql("Training_database_backup", con, if_exists="replace")

exception_table = pd.read_sql_query("SELECT * from Exceptions", con)
exception_table.to_sql("Exceptions_backup", con, if_exists="replace")

     

#Data / model Preparation ---------------------
data = data.dropna()
threshold_accuracy = 0.9
threshold_precision = 0.8

#Extract columns from database and read 
columns = data.columns.values.tolist()
columns.pop(0)
print("Please tell us what do you want to do:")

#load list of functions, levels and exception words
exception_words_functions = exception_table["Excep_Functions"].tolist()
exception_words_automation_role = exception_table["Excep_Automation"].tolist()
exception_words_level = exception_table["Exception_Level"].tolist()
list_of_functions = ["Finance","Human_resources","Tech","Legal","Sales","Marketing","Communications","SC_operations","Field_operations","Field_operations"]
list_of_levels = ["CEO","C_level","SVP_VP","Director","Manager","Supervisor","Executive_decisions"]


def run_and_checkaccuracy():
    
    global client_data_full   


    client_data_full = pd.read_sql_query("SELECT * from Client_data_output", con)
    client_data_full.to_sql("Client_data_output_backup", con, if_exists="replace")
    client_data_full = client_data_full.drop(columns = ["index"], errors = 'ignore')
    print(client_data_full.head())
     
    #Split train test and run the model for each column

    
    for column_selected in columns:
        
        X_train, X_test, y_train, y_test = train_test_split(data['Description'],data[column_selected], test_size=0.10, random_state=42)
        from sklearn.feature_extraction.text import CountVectorizer
    
    #ELIMINATE NON-MEANINGFUL WORDS DEPENDING ON THE TYPE OF ASSESMENT---------------------------------- 
        
    # if we are predicting functions, we ignore very common words
        if column_selected in list_of_functions:
            count_vector = CountVectorizer(min_df = 5, max_df =0.40,stop_words=exception_words_functions)
            print("Category: Function")
    # if we are predicting level, we ignore very common words
        elif column_selected in list_of_levels:
            count_vector = CountVectorizer(min_df = 10, max_df =0.40,stop_words=exception_words_level)
            print("Category: Level")
        else:
            count_vector = CountVectorizer(min_df =5,stop_words=exception_words_automation_role)
            print("Category: Automation_Potential")
        training_data = count_vector.fit_transform(X_train.values.astype('U'))
        testing_data = count_vector.transform(X_test)
    
    #CALCULATING THE FREQUENCY OF MOST COMMON WORDS---------------------------------- 
        
        sum_words = training_data.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in     count_vector.vocabulary_.items()]    
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        
    
    #TRAIN NAIVE BAYES ALGO    
        
        print(training_data.shape)
        from sklearn.naive_bayes import MultinomialNB
        naive_bayes = MultinomialNB(alpha = 0.001)
        naive_bayes.fit(training_data, y_train)
        predictions = naive_bayes.predict(testing_data)
        
    #CALCULATE PRECISION METRICS
    
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        if accuracy_score(y_test, predictions)< threshold_accuracy or precision_score(y_test, predictions) < threshold_precision:
            print('Model is not well trained for: '+column_selected+':')
            print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
            print('Precision score: ', format(precision_score(y_test, predictions)))
            print('Recall score: ', format(recall_score(y_test, predictions)))
            print('F1 score: ', format(f1_score(y_test, predictions)))
            training_results = X_test 
            training_results = training_results.to_frame()
            training_results['Real_data'] = y_test
            training_results['Prediction'] = predictions
            mismatch = training_results[training_results.Prediction != training_results.Real_data]
            print(mismatch)
            print(words_freq[:30])
        else:
            print(column_selected + ' is OK')
    
        #Creating a dataframe to store the predictions
        predictions_dataframe = pd.DataFrame({'Pred': predictions})
        #X input for the client file - description of Job Titles
        X_real = client_data_full['Description'].values.astype('U')
    
    #Naive Bayes Algo predicting results on real data (client data)
       
        real_data = count_vector.transform(X_real)
        predictions = naive_bayes.predict(real_data)
        predictions_dataframe = pd.DataFrame({'Pred': predictions})
    
    #Storing predictions in the client database
    
        
        client_data_full[column_selected] = predictions_dataframe
    
    #Automation potential calculation
    client_data_full['Automation Potential'] =  'No potential'
    client_data_full['Automation Potential'] =  np.where(client_data_full['Check_multiple_systens']>0, 'Assist', client_data_full['Automation Potential'])
    client_data_full['Automation Potential'] =  np.where(client_data_full['Periodic_activities_structured']>0, 'Some Autonomy',client_data_full['Automation Potential'])
    client_data_full['Automation Potential'] =  np.where(client_data_full['Data_calculations']>0 , 'Partially Autonomous',client_data_full['Automation Potential'])
    client_data_full['Automation Potential'] =  np.where(client_data_full['Structured']>0, 'Mostly Autonomous', client_data_full['Automation Potential'])
    client_data_full['Automation Potential'] =  np.where(client_data_full['Executive_decisions']>0 , 'No potential', client_data_full['Automation Potential'])
    client_data_full['Automation Potential'] =  np.where(client_data_full['Neural']< 1, 'No potential', client_data_full['Automation Potential'])
    # client_data_full['Automation Potential'] =  np.where(client_data_full['No_Potential_Exception']> 0, 'No potential', client_data_full['Automation Potential'])

    #Function Calculation
    client_data_full['Function'] = 'Not Classified'
    function = ['Other_G&A','Field_operations','Communications','SC_operations','Legal','Tech','Human_resources','Sales','Marketing','Finance']
    for x in function:
        client_data_full['Function'] =  np.where(client_data_full[x]>0, x, client_data_full['Function'])
    
    #Level Calculation
    client_data_full['Level'] = 'Non managerial'
    level = ['C_level','SVP_VP','Director','Manager', 'Supervisor']
    for x in level:
        client_data_full['Level'] =  np.where(client_data_full[x]>0, x, client_data_full['Level'])
    
    
    client_data_full['Automation Potential'] =  np.where(client_data_full['Function']== "Not Classified", 'No potential', client_data_full['Automation Potential'])
    client_data_full['Automation Potential'] =  np.where(client_data_full['Director']>0, 'No potential', client_data_full['Automation Potential'])
    client_data_full['Automation Potential'] =  np.where(client_data_full['C_level']>0, 'No potential', client_data_full['Automation Potential'])
    client_data_full['Automation Potential'] =  np.where(client_data_full['SVP_VP']>0, 'No potential', client_data_full['Automation Potential'])
    
    
    
    #Export data to client spreadhsheet
    
    print("Complete!")
    
#    client_data_full = client_data_full.drop(columns=['level_0'])
    client_data_full.to_sql("Client_data_output", con, if_exists="replace")
    
    query = pd.read_sql_query("SELECT * from Tableau_output", con)
    query.to_csv('Tableau_output.csv')



def Add_new_company():
    company_list = pd.read_sql_query("SELECT * from Company", con)
    company_list.to_sql("Company_backup", con, if_exists="replace")
    
    print("Before you add a company, please make sure it is not contained in the following list:")
    print(company_list.dropna())
    inp = input("Do you stil want to add a new client? Yes/No :")
    if inp == "Yes":
        col1 = input("What is the name of your client?")
        col2 = input("What is the PwC Vertical?")
        col3 = input("What is the Sector?")
        company_list = company_list.dropna()
        col4 = company_list["ID"].max()+1
        a_row = pd.Series([col1, col2,col3,col4],index = ["Company", "PwC Vertical", "Sector","ID"])
        print(a_row)
        row_df = pd.DataFrame([a_row])   
        print(row_df)
        company_list = pd.concat([row_df, company_list.dropna()], ignore_index=True)
        
        print(company_list)
        company_list.to_sql("Company", con, if_exists="replace")

def Download_import_template():
    from datetime import datetime
    extract_list = pd.read_sql_query("SELECT * from Extract", con)
    extract_list.to_sql("Extract_backup", con, if_exists="replace")
    print(extract_list.dtypes)
    print(type(extract_list))
    focus = input("Please write the focus of employee list the extract (e.g. All, SG&A )")
    now = datetime.now()
    source_data = input("Please write the source of the data (e.g. Linkedin, Client)")
    company = input("Please select a company)")
    type_data = "Employee Headcount"
    ID = 1
    print(type(extract_list))
    columnID = extract_list["ID"].astype('int')
    ID = columnID.max()+1
    row_df = pd.DataFrame(np.array([[focus, now,type_data,ID,company,source_data]]), columns=["Focus of the Extract", "Extraction Date", "Type","ID","Company","Source"])
    extract_list = pd.concat([row_df, extract_list], ignore_index=True)
    extract_list = extract_list
    client_data_full = pd.read_sql_query("SELECT * from Client_data_output", con)
    name_of_file="AIOutsidein_Extract_ID_"+str(ID)+"_Company_"+company+".csv"
    client_data_full.filter(like='xsdsasdfsdxassaas', axis=0).to_csv(name_of_file)
    extract_list.drop(columns=['level_0'],errors="ignore")
    extract_list = extract_list[["Focus of the Extract",'Extraction Date',"Type","ID","Company","Source"]]
    extract_list = extract_list.astype('str')    
    extract_list.to_sql("Extract", con, if_exists="replace")


def Upload_data():
    ID2 = input("What is the extract number?")
    ID3 = input("What is the company name?")

    name_of_file="AIOutsidein_Extract_ID_"+str(ID2)+"_Company_"+ID3+".csv"
    print(name_of_file)
    new_data = pd.read_csv(name_of_file)
    new_data["EXTRACT_ID"] = ID2
    new_data.to_sql("Client_data_output", con, if_exists="append")
   
    


#USER INTERFACE
options = ["1. Run the model and check accuracy", "2. Add new company to the model", "3. Add new employee database extract to the model and download template","4. Upload Template"]

# Print out your options
for i in range(len(options)):
    print(str(i+1) + ":", options[i])

# Take user input and get the corresponding item from the list
inp = int(input("Enter a number: "))
if inp in range(1, 5):
    inp = options[inp-1]
else:
    print("Invalid input!")

if inp == "1. Run the model and check accuracy":
    run_and_checkaccuracy()
    
elif inp == "2. Add new company to the model":
    Add_new_company()
    
elif inp == "3. Add new employee database extract to the model and download template":
    Download_import_template()

elif inp == "4. Upload Template":
    Upload_data()
    
    


con.close()
# In[ ]:




# In[ ]:



