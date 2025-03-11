import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle

# Load CSV files using relative paths
def laad_csv_bestand():
    base_path = os.path.dirname(os.path.abspath(__file__))
    forecast_path = os.path.join(base_path, "datasets", "forecast.csv")
    sunset_path = os.path.join(base_path, "datasets", "sunrise-sunset.xlsx")
    df_path = os.path.join(base_path, "datasets", "df.csv")
    
    forecast = pd.read_csv(forecast_path)
    sunset = pd.read_excel(sunset_path)
    df = pd.read_csv(df_path)
    
    return forecast, sunset, df

def laden_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    bestand_pad = os.path.join(base_path, "best_model.pkl")
    
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

    # Load and prepare the data for training
    X = df[['Uur', 'humidity_relative', 'cloudiness', 'temp', 'pressure', 'Opkomst', 'Ondergang']]
    y = df['kwh']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Retrieve the model using the pickle module
    model = laden_model()

    # Train the model and evaluate it
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)

    # Ensure the columns are in the correct order for the model
    X_forecast = forecast_merged[['Uur', 'humidity_relative', 'cloudiness', 'temp', 'pressure', 'Opkomst', 'Ondergang']]
    y_forecast = model.predict(X_forecast)

    # Print results with date and hour
    print("\nPredictions:")
    for prediction, date, hour in zip(y_forecast, forecast_merged['Datum'], forecast_merged['Uur']):
        formatted_date = pd.to_datetime(date).strftime('%Y-%m-%d')
        print(f"Prediction for {formatted_date} at {hour}h: {prediction:.2f} kWh")

    # Create a datetime column for plotting
    forecast_merged['Datetime'] = pd.to_datetime(forecast_merged['Datum'].astype(str)) + pd.to_timedelta(forecast_merged['Uur'], unit='h')

    # Plot the predictions with improved formatting
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_merged['Datetime'], y_forecast, marker='o', linestyle='-', linewidth=2)
    plt.xlabel("Datetime")
    plt.ylabel("Predicted kWh")
    plt.title("Solar Energy Predictions")
    plt.grid(True)

    # Format x-axis so that the hour is displayed with 'h'
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %Hh'))
    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

    # Create 'img' directory if it doesn't exist and save the plot
    base_path = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_path, "img")
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, "predictions.png")
    plt.savefig(img_path)
    print(f"\nPlot saved to {img_path}")
    plt.show()

# Main function that calls the above functions
def main():
    forecast, sunset, df = laad_csv_bestand()
    toepassen_model(forecast, sunset, df)

if __name__ == '__main__':
    main()
