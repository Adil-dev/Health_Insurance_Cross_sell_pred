import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def prep_func(df_e):
    df = df_e
    va = {'> 2 Years': 2, '1-2 Year': 1, '< 1 Year': 0}
    gen = {'Male' : 0, 'Female' : 1}
    vg = {'Yes' : 1, 'No' : 0}
    df['Vehicle_Age'] = df['Vehicle_Age'].map(va)
    df['Gender'] = df['Gender'].map(gen)
    df['Vehicle_Damage'] = df['Vehicle_Damage'].map(vg)

    num_feat = ['Age', 'Vintage', 'Annual_Premium']

    df = df.query('Annual_Premium <= 100000')

    cat_feat = [
    'Gender', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
    'Driving_License', 'Policy_Sales_Channel', 'Region_Code']

    scl = MinMaxScaler()

    num_scl = pd.DataFrame(scl.fit_transform(df[num_feat]))
    num_scl.index = df[num_feat].index
    num_scl.columns = df[num_feat].columns
    X = pd.concat([num_scl, df[cat_feat]], axis=1)
    
    return X