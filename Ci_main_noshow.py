# Importing all usefull libraries
import pandasql as ps
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.externals import joblib
from sklearn import preprocessing
from numpy import timedelta64, nan, append
import pickle
import time
import sys
import datetime
from dateutil.relativedelta import relativedelta
from gc import collect
import boto3
import json
import joblib
from sklearn.metrics import log_loss,confusion_matrix, r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, auc, accuracy_score, precision_score
from tabulate import tabulate
import random

# Importing libraries for the modelling
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Python script for confusion matrix creation. 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

# Own Libraries
#import ci_utils
# from Ci_preprocessing import preprocessing_noshow3

# Librerias
import pandas as pd
import pypyodbc # pip install pypyodbc


def preprocessing_noshow_3(df):
    df.columns = df.columns.str.lower()
    df = df[[#'id_paciente',
             'fecha_cita','hora_cita','no_show','canal','edad','sexo','id_medico'
             ,'duracion_cita','especialidad_medica','ubicacion'
             ,'fecha_programacion','hora_programacion','id_paciente',
                'codigo_ecivil',
                'codigo_profesion',
                'garante',
                'distrito',
                'cant_adic',
                'app5_noshow',
                'prior_app',
                'prior_noshow',
                #'ind_noshow',
                'prior_asistencias',
                'new_patient',
                'cant_telef',
                'prior_anull' ]]
    # For dates
    df['fecha_cita'] = pd.to_datetime(df['fecha_cita'], errors = 'coerce')
    df['fecha_programacion'] = pd.to_datetime(df['fecha_programacion'], errors = 'coerce')
    df['hora_cita'] = pd.to_datetime(df['hora_cita'], errors = 'coerce')
    df['hora_programacion'] = pd.to_datetime(df['hora_programacion'], errors = 'coerce')
    df['difference_in_dates'] = (df['fecha_cita'] - df['fecha_programacion']).dt.days.abs() 
    df['difference_in_hours'] =  (df['hora_cita'] - df['hora_programacion']).abs()/np.timedelta64(1,'h')
    df['difference_in_minutes'] =  (df['hora_cita'] - df['hora_programacion']).abs()/np.timedelta64(1,'m')
    df['difference_in_seconds'] =  (df['hora_cita'] - df['hora_programacion']).abs()/np.timedelta64(1,'s')
    df['cita_month'] = df['fecha_cita'].dt.month
    df['cita_day'] = df['fecha_cita'].dt.day
    dayOfWeek={0:'Lunes', 1:'Martes', 2:'Miercoles', 3:'Jueves', 4:'Viernes', 5:'Sabado', 6:'Domingo'}
    df['cita_day_of_week'] = df['fecha_cita'].dt.dayofweek.map(dayOfWeek)
    # Add new columns
    df['no_show_binary'] = 0
    df.loc[df['no_show'] == 'SI', ['no_show_binary']] = 1
    df['pm_cita'] = 0
    df.loc[df['hora_cita'] > '12:00:00', ['pm_cita']] = 1
    df.loc[df['edad'] == 'No disponible', ['edad']] = float('NAN')
    # Convert edad to all numbers
    df['edad'] = pd.to_numeric(df['edad'])
    df['new_patient'] = pd.to_numeric(df['new_patient'])
    df['cant_adic'] = pd.to_numeric(df['cant_adic'])
    df['app5_noshow'] = pd.to_numeric(df['app5_noshow'])
    #df['prior_noshow'] = pd.to_numeric(df['prior_noshow'])
    #df['prior_app'] = pd.to_numeric(df['prior_app'])
    #ind_noshow
    return df

# Parametros de bases
# cnxn = pypyodbc.connect("Driver={SQL Server};"
#                         "Server=172.24.144.48;"
#                         "Database=NDMC_ISOFT;")
# Query no show
query_val = """ SELECT  ID_CITA,
                        ID_PACIENTE,
                        FECHA_CITA,
                        HORA_CITA,
                        YEAR(FECHA_CITA) AS YEAR_CITA,
                        CANAL,
                        NO_SHOW,
                        EDAD,
                        SEXO,
                        ID_MEDICO,
                        FPER.NOMBRE_CORTO as NOMBRE_MEDICO,
                        DURACION_CITA,
                        ESPECIALIDAD_MEDICA,
                        UBICACION,
                        FECHA_PROGRAMACION,
                        HORA_PROGRAMACION,
                        CASE WHEN YEAR(FECHA_CITA) = YEAR(FECHA_PROGRAMACION)  AND MONTH(FECHA_CITA) = MONTH(FECHA_PROGRAMACION) AND DAY(FECHA_CITA) = DAY(FECHA_PROGRAMACION)
                        THEN 'ESPONT' ELSE 'AMB' END AS TIPO_CITA_2,
                        ROW_NUMBER() OVER (PARTITION BY ID_PACIENTE ORDER BY FECHA_CITA DESC) ENU,
                        CASE WHEN lower(NOTA) LIKE '%prev%' OR lower(NOTA) LIKE '%chequeo%' or lower(NOTA) LIKE '%*eps*%'
                        THEN 'PREVENTIVO' ELSE 'AMBULATORIO' END AS TIPO_CITA,
                        CLI.TELEFONO1,
                        CLI.TELEFONO2,
                        --RTRIM(LTRIM(CLI.APELLIDO1)) + ' ' + RTRIM(LTRIM(CLI.APELLIDO2)) + ' ' + RTRIM(LTRIM(CLI.NOMBRE)) AS NOMBRE,
                        RTRIM(LTRIM(CLI.NOMBRE)) + ' ' +  RTRIM(LTRIM(CLI.APELLIDO1)) AS NOMBRE,
                        CLI.CODIGO1 AS DNI,
                        CODIGO_ECIVIL,
                        CODIGO_PROFESION,
                        GARANTE,
                        DISTRITO

                FROM NDMC_ISOFT.DBO.DATA_NO_SHOW_MIT TB_MODEL
                LEFT JOIN NDMC_ISOFT.DBO.CAGENDAS AGE ON AGE.N_SOLIC = TB_MODEL.ID_CITA
                LEFT JOIN NDMC_ISOFT.DBO.CLIENTES CLI ON CLI.CODIGO_CLIENTE = TB_MODEL.ID_PACIENTE 
                LEFT JOIN NDMC_ISOFT.DBO.FPERSONA FPER ON FPER.CODIGO_PERSONAL = TB_MODEL.ID_MEDICO
                WHERE   FECHA_CITA <= (CASE WHEN DATEPART(WEEKDAY,GETDATE()) = 7 
                                            THEN DATEADD(DAY, 2, CONVERT(date,GETDATE())) 
                                            ELSE DATEADD(DAY, 1, CONVERT(date,GETDATE())) END) AND
                        FECHA_CITA >=  DATEADD(DAY, -365, CONVERT(date,GETDATE())) """
                    
# df_test = pd.read_sql_query(query_val, cnxn)

df_test = ""
df_test = pd.read_csv('DF_noshow_smp.csv', sep=';')

# Query para obtener fecha de gestion
query_date = """SELECT CASE WHEN DATEPART(WEEKDAY,GETDATE()) = 7 
                            THEN DATEADD(DAY, 2, CONVERT(date,GETDATE())) 
                            ELSE DATEADD(DAY, 1, CONVERT(date,GETDATE())) END AS FECHA_GESTION, 
                       DATEADD(DAY, -365, CONVERT(date,GETDATE())) AS FECHA_RELACION """

# df_date = pd.read_sql_query(query_date, cnxn)
df_date= df_test
################################################################################################################################################################################
################################################################################################################################################################################


# filtros iniciales. Ambulatorio y espontaneos.
df_ns = df_test
df_ns_new = df_ns[(df_ns.tipo_cita == 'AMBULATORIO') & (df_ns.canal != 'Espontaneo') & (df_ns.tipo_cita_2 != 'ESPONT')]
# df_ns_new = df_ns_new.replace('[Sin Descripción]', None)
df_ns_new = df_ns_new.replace({'[Sin Descripción]': None})

# last date appointment per patient
df_ult_fech = df_ns_new[df_ns_new.enu == 1][["id_paciente","fecha_cita"]]
df_ult_fech.columns = ["id_paciente", "ult_fecha"]

# Difference in dates
df_ns_2 = pd.merge(df_ns_new,df_ult_fech, on = 'id_paciente', how ='left')
df_ns_2["dif_dias"] = (pd.to_datetime(df_ns_2['ult_fecha'], errors = 'coerce') - pd.to_datetime(df_ns_2['fecha_cita'], errors = 'coerce')).dt.days.abs()

# Data train 
df_unique  = df_ns_2[(df_ns_2.enu == 1) & (df_ns_2.fecha_cita == df_date["fecha_gestion"][0])]

# Data adicionales
df_adicio  = df_ns_2[(df_ns_2.enu != 1) & (df_ns_2.fecha_cita == df_date["fecha_gestion"][0]) & (df_ns_2.dif_dias == 0)]

# Data history
df_history = df_ns_2[(df_ns_2.enu != 1) & (df_ns_2.dif_dias <= 365) & (df_ns_2.dif_dias != 0)]
df_history["cont"] = df_history.groupby('id_paciente').cumcount() + 1


# Citas adicionales: mismo dia
df_app_unique = df_adicio[['id_paciente','enu']].groupby(['id_paciente']).count()
df_app_unique.reset_index(inplace=True)
df_app_unique.columns = ["id_paciente","cant_adic"]

df_ult_5 = df_history[(df_history.cont <= 5) & (df_history.no_show == "SI")]
df_ult_5 = df_ult_5[['id_paciente','enu']].groupby(['id_paciente']).count()
df_ult_5.reset_index(inplace=True)
df_ult_5.columns = ["id_paciente","app5_noshow"]

####################################################################################
df_ult_4 = df_history[(df_history.cont <= 4) & (df_history.no_show == "SI")]
df_ult_4 = df_ult_4[['id_paciente','enu']].groupby(['id_paciente']).count()
df_ult_4.reset_index(inplace=True)
df_ult_4.columns = ["id_paciente","app4_noshow"]
segm_ult_4 = df_ult_4[df_ult_4.app4_noshow >= 2]
segm_ult_4["app4_noshow"] = 1
####################################################################################

df_app = df_history[['id_paciente', "enu"]].groupby(['id_paciente']).count()
df_app.reset_index(inplace=True)
df_app.columns = ["id_paciente","prior_app"]

df_noshow = df_history[df_history.no_show == 'SI'][["id_paciente","no_show"]].groupby(['id_paciente']).count()
df_noshow.reset_index(inplace=True)
df_noshow.columns = ["id_paciente","prior_noshow"]

df_conso = pd.merge(df_app,df_noshow, on = 'id_paciente', how ='left')
df_conso['ind_noshow'] = round(df_conso["prior_noshow"]/df_conso["prior_app"],2)
df_conso['prior_noshow'].fillna(0, inplace=True)
df_conso['ind_noshow'].fillna(0, inplace=True)

# Cantidad de asistencias en los últimos 12 meses
df_conso['prior_asistencias'] = df_conso.prior_app - df_conso.prior_noshow

df_new = pd.DataFrame()
df_new["id_paciente"] = df_history.id_paciente.unique()
df_new["new_patient"] = 0
# df_final['new_patient'].replace(True,1)
# df_final['ind_noshow'].fillna(0, inplace=True)

df_contacto = df_unique[["id_paciente", "telefono1", "telefono2"]]
df_contacto.fillna(value=pd.np.nan, inplace=True)
df_contacto['telefono1'].fillna(0, inplace=True)
df_contacto['telefono2'].fillna(0, inplace=True)
df_contacto.loc[df_contacto.telefono1 != 0, 'telefono1'] = 1
df_contacto.loc[df_contacto.telefono2 != 0, 'telefono2'] = 1
# cantidad de numeros telefónicos
df_contacto["cant_telef"] = df_contacto.telefono1 + df_contacto.telefono2
df_contacto = df_contacto[['id_paciente','cant_telef']]

# Parametros de bases
cnxn = pypyodbc.connect("Driver={SQL Server};"
                        "Server=172.24.144.48;"
                        "Database=NDMC_ISOFT;")

# Query no show
query_anul = """ SELECT CAGE_ANUL.CODIGO_CLIENTE AS ID_PACIENTE,
                        FECHA_EMISION,
                        FECHA,
                        FECHA_ANUL
                    FROM [NDMC_ISOFT].[dbo].[CAGENDAS_ANUL] CAGE_ANUL LEFT JOIN 
                    [NDMC_ISOFT].[dbo].[CLIENTES] CLI ON CLI.CODIGO_CLIENTE = CAGE_ANUL.CODIGO_CLIENTE
                    WHERE   FECHA <= (CASE WHEN DATEPART(WEEKDAY,GETDATE()) = 7 
                                            THEN DATEADD(DAY, 2, CONVERT(date,GETDATE())) 
                                            ELSE DATEADD(DAY, 1, CONVERT(date,GETDATE())) END) AND
                            FECHA >=  DATEADD(DAY, -365, CONVERT(date,GETDATE())) """

df_anul = pd.read_sql_query(query_anul, cnxn)
df_anul.fillna(value=pd.np.nan, inplace=True)
df_anul = df_anul.dropna()
df_anul = df_anul.reset_index(drop=True)

df_anul_2 = pd.merge(df_anul,df_unique[["id_paciente","fecha_cita"]], on = 'id_paciente', how ='inner')
df_anul_2["dif_dias"] = (pd.to_datetime(df_anul_2['fecha_cita'], errors = 'coerce') - pd.to_datetime(df_anul_2['fecha'], errors = 'coerce')).dt.days.abs()
df_anul_2 = df_anul_2[df_anul_2.dif_dias <= 365]

# Cantidad de anulados
df_anulados = df_anul_2[["id_paciente","fecha_emision"]].groupby(['id_paciente']).count()
df_anulados.reset_index(inplace=True)
df_anulados.columns = ["id_paciente","prior_anull"]

df_final_0 = pd.merge(df_unique,df_app_unique, on = 'id_paciente',how='left')
df_final_1 = pd.merge(df_final_0,df_ult_5, on = 'id_paciente',how='left')
df_final_2 = pd.merge(df_final_1,df_conso, on = 'id_paciente',how='left')
df_final_3 = pd.merge(df_final_2,df_new, on = 'id_paciente',how='left')
df_final_4 = pd.merge(df_final_3,df_contacto, on = 'id_paciente',how='left')
df_final_5 = pd.merge(df_final_4,df_anulados, on = 'id_paciente',how='left')

df_final_5['cant_adic'].fillna(0, inplace=True)
df_final_5['app5_noshow'].fillna(0, inplace=True)
df_final_5['prior_app'].fillna(0, inplace=True)
df_final_5['prior_noshow'].fillna(0, inplace=True)
df_final_5['ind_noshow'].fillna(0, inplace=True)
df_final_5['prior_asistencias'].fillna(0, inplace=True)
df_final_5['new_patient'].fillna(1, inplace=True)
df_final_5['prior_anull'].fillna(0, inplace=True)

################################################################################################################################################################################
################################################################################################################################################################################

# Preprocesing
df = preprocessing_noshow_3(df_final_5)
#df = preprocessing_noshow(df_10_enero)

# Feature Selection
# Identify the target
target = 'no_show_binary'
# Identify the variables to exclude
exclude = [
    'id_paciente',
    'no_show',
    'fecha_cita',
    'fecha_programacion',
    'hora_cita',
    'hora_programacion'
]
# Identify the list of targets
target_list = [
    target
]
# Append target_list to exclude
for item in target_list:
    exclude.append(item)
# Identify categorical features
categorical_variables = [
    'especialidad_medica', 
    'garante',
    'distrito',
    'ubicacion', 
    'canal',
    'cita_day_of_week'
]
numeric_categorical_variables = [
    'id_medico',
    'sexo',
    'new_patient',
    'cita_month', 
    'cita_day'
]
# Append Categorical Variables to Exclude list
for var in categorical_variables:
    exclude.append(var)
# Using the dataframe columns, excluded columns, and target columns to return a list of included columns for training
include = []
for column in df.columns:
    if column not in exclude:
        if column not in target_list:
            include.append(column)
#Thresholding Logic for Features
# Natural Number
natural_number = 42.9

# Natural Rate
natural_rate = df['no_show_binary'].value_counts(normalize = True)[1]

# Threshold 
thresh = natural_number / natural_rate

# Make threshold cuts on all categorical_variables with value counts less than the threshold (this ensures that they have statistical significance)
for var in categorical_variables:
    # Setting threshold for the variable
    var_thresh = []
    var_counts = df[var].value_counts()
    var_counts.to_dict()

    for key in var_counts.keys():
        if var_counts[key] < thresh:
            var_thresh.append(key)
    df[var].replace(var_thresh, 'not_enough', inplace = True)  # Label them in the 'NOT_ENOUGH' group

# Make threshold cuts on all numeric variables with value counts less than the threshold (this ensures that they have statistical significance)
for var in numeric_categorical_variables:         # only contains numeric values right now
    # Setting threshold for the variable
    var_thresh = []
    var_counts = df[var].value_counts()
    var_counts.to_dict()
    for key in var_counts.keys():
        if var_counts[key] < thresh:
            var_thresh.append(key)
    df[var].replace(var_thresh, .0001, inplace = True)  # Label them in the .0001 group

#Label Encoding Categorical Variables
# Fill missing values
df[categorical_variables] = df[categorical_variables].fillna('missing_value')

# Changing the data type of category list in the dataframe
for col in categorical_variables:
    df[col] = pd.Categorical(df[col])

# Change the dtype of the categorical variables in the dataframe
for col in categorical_variables:
    df[col].astype('category')

# Loop through categorical variables creating label encoded columns
for var in categorical_variables:
    # String names for the loop
    encoder_name = var + '_encoder'
    column_title = var + '_code'
    
    # Initialize encoder
    encoder = preprocessing.LabelEncoder()
    
    # Create columns with coded variables
    df[column_title] = encoder.fit_transform(df[var])
    
    # Append columns to include
    include.append(column_title)
    
#Check that All Variables in include are numeric
df.replace(to_replace = '#VALUE!', value = np.nan, inplace = True)
for col in include:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col])

        
df_new_10 = df

test_columns = df_new_10[include]
test_target = df_new_10[target]

# Pipeline
try:
    from sklearn.impute import SimpleImputer as Imputer
except ImportError:
    from sklearn.preprocessing import Imputer

imputer = joblib.load("Train_imputer_noshow.sav")
exported_pipeline = joblib.load("Train_model.sav")

testing_features_10 = imputer.transform(test_columns)
results_10 = exported_pipeline.predict(testing_features_10)

df_new_10['regress'] = results_10
# Create pred_target column
# Initial value for column is 3 (max (cortes[3]) cut is the maximum value of the table)
df_new_10['pred_target'] = 1

# If below the cortes 0, set the value to 0
cortes = pd.read_csv('Train_corte_noshow_predict.csv')

df_new_10.loc[df_new_10['regress'] < cortes["corte"][0], 'pred_target'] = 0

# Create correct column -> Set to True if the TARGET value is equal to the pred_target VALUE
df_new_10['correct'] = df_new_10[target] == df_new_10['pred_target']

# Confusion Matrix
results = confusion_matrix(df_new_10['no_show_binary'], df_new_10['pred_target'])

# Print results
# print ('Confusion Matrix :')
# print (results)
# print ('Accuracy Score :',round(accuracy_score(df_new_10['no_show_binary'], df_new_10['pred_target'])*100,1) )
# print ('Report : ')
# print (classification_report(df_new_10['no_show_binary'], df_new_10['pred_target']) )

################################################################################################################################################################################
################################################################################################################################################################################
# Query fech
query_time = """ SELECT CASE WHEN DATEPART(WEEKDAY,GETDATE()) = 7 
                        THEN DATEADD(DAY, 2, CONVERT(date,GETDATE())) 
                        ELSE DATEADD(DAY, 1, CONVERT(date,GETDATE())) END AS TIME"""

query_time = pd.read_sql_query(query_time, cnxn)

df_predict_day = df_new_10[["id_paciente", "fecha_cita","hora_cita","ubicacion","pred_target"]]
df_cruce_contact = df_final_5[["id_paciente","id_cita","telefono1","telefono2","nombre","dni","nombre_medico","especialidad_medica"]]
df_predict_final = pd.merge(df_predict_day,df_cruce_contact, on = 'id_paciente', how ='left')
df_predict_final = df_predict_final[["id_cita","id_paciente","dni","nombre","telefono1","telefono2",
                                    "fecha_cita","hora_cita","especialidad_medica","nombre_medico","ubicacion","pred_target"]]


#df_predict_final["fecha_cita"] = pd.to_datetime( df_predict_final["hora_cita"].dt.time.apply(str) +' '+ df_predict_final["fecha_cita"].dt.date.apply(str))
#del df_predict_final['hora_cita']

df_predict_final['fecha_cita'] = df_predict_final["fecha_cita"].dt.date
df_predict_final['hora_cita'] = df_predict_final["hora_cita"].dt.time


# Enviar para Contact Center
#df_predict_final.to_excel (r'lista_noshow_contact_'+query_time.time[0]+'.xlsx', index = None, header=True)

# Enviar para Atento
####################################################################################

aasa = pd.merge(df_predict_final,segm_ult_4, on = 'id_paciente', how ='left')
aasa["app4_noshow"].fillna(0, inplace=True)
aasa["pred_target_2"] = aasa.pred_target + aasa.app4_noshow
aasa["pred_target_2"][aasa.pred_target_2 > 1] = 1

df_predict_final_ns = aasa[aasa.pred_target_2 == 1]

del df_predict_final_ns['pred_target']
del df_predict_final_ns['pred_target_2']
del df_predict_final_ns['app4_noshow']

####################################################################################

## create dnis fakes

rand_dnis = list()
for i in range(df_predict_final_ns["id_cita"].size):
    rand_dnis.append(random.randrange(10000000, 99999999))
    
df_predict_final_ns["dni"] = rand_dnis

df_predict_final_ns.to_excel (r'lista_noshow_'+query_time.time[0]+'.xlsx', index = None, header=True)

df_predict_final_ns['id_cita'] = df_predict_final_ns['id_cita'].astype(int)
df_predict_final_ns.to_excel (r'lista_noshow_'+query_time.time[0]+'_call.xlsx', index = None, header=True)




