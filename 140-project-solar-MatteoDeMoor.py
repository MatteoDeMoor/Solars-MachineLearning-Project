import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle

# Load CSV files
def laad_csv_bestand():
    forecast = pd.read_csv("datasets/forecast.csv")
    sunset = pd.read_excel("datasets/sunrise-sunset.xlsx")
    df = pd.read_csv("datasets/df.csv")

    return forecast, sunset, df

def laden_model():
    # The path to the saved .pkl file in the current directory
    bestand_pad = "best_model.pkl"

    # Load the saved model
    with open(bestand_pad, 'rb') as f:
        geladen_model = pickle.load(f)

    return geladen_model

# Transform data and train the model
def toepassen_model(forecast, sunset, df):
    # Prepare the forecast dataset
    forecast['timestamp'] = pd.to_datetime(forecast['timestamp'])
    forecast['Datum'] = pd.to_datetime(forecast['timestamp'].dt.date)
    forecast['Uur'] = forecast['timestamp'].dt.hour
    forecast.drop('timestamp', axis=1, inplace=True)

    # Prepare the sunset dataset
    sunset['Datum'] = pd.to_datetime(sunset['datum'])
    sunset['Opkomst'] = pd.to_datetime(sunset['Opkomst'], format='%H:%M:%S').dt.hour * 60 + pd.to_datetime(sunset['Opkomst'], format='%H:%M:%S').dt.minute
    sunset['Op ware middag'] = pd.to_datetime(sunset['Op ware middag'], format='%H:%M:%S').dt.hour * 60 + pd.to_datetime(sunset['Op ware middag'], format='%H:%M:%S').dt.minute
    sunset['Ondergang'] = pd.to_datetime(sunset['Ondergang'], format='%H:%M:%S').dt.hour * 60 + pd.to_datetime(sunset['Ondergang'], format='%H:%M:%S').dt.minute
    sunset.drop('datum', axis=1, inplace=True)

    # Merge the forecast and sunset dataframes on the 'Datum' column
    forecast_merged = forecast.merge(sunset, on='Datum', how='left')

    # Load and prepare the data
    X = df[['Uur', 'humidity_relative', 'cloudiness', 'temp', 'pressure', 'Opkomst', 'Ondergang']]
    y = df['kwh']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Retrieve the model using the pickle module
    model = laden_model()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)

    # Ensure the columns are in the correct order for the model
    X_forecast = forecast_merged[['Uur', 'humidity_relative', 'cloudiness', 'temp', 'pressure', 'Opkomst', 'Ondergang']]
    y_forecast = model.predict(X_forecast)

    # Print results with date and hour
    print("\nVoorspellingen:")
    for i, (prediction, date, hour) in enumerate(zip(y_forecast, forecast_merged['Datum'], forecast_merged['Uur']), 1):
        # Format the date to the correct format
        formatted_date = pd.to_datetime(date).strftime('%Y-%m-%d') 

        # Print the prediction
        print(f"Voorspelling voor {formatted_date} om {hour} uur: {prediction:.2f} kWh")

# Main function that calls the above functions
def main():
    forecast, sunset, df = laad_csv_bestand()
    toepassen_model(forecast, sunset, df)

if __name__ == '__main__':
    main()
