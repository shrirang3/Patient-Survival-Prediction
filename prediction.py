import joblib
model=joblib.load('./model.pkl')
from sklearn.preprocessing import MinMaxScaler

def ScaleData(df):
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    return df   


def get_prediction(model, df):
    prediction = model.predict(df)
    return prediction