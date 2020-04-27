import pandas as pd
import numpy as np

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


