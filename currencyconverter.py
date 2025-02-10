import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
import requests
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import matplotlib.dates as mdates
import tkinter
from PIL import Image, ImageTk
from tkinter import ttk
from forex_python.converter import CurrencyRates, CurrencyCodes
from forex_python.bitcoin import BtcConverter
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import numpy as np
import json
from fredapi import Fred
import warnings
# Initialize required objects
btc_converter = BtcConverter() 
c = CurrencyRates()
codes = CurrencyCodes()

# Define the LSTM model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# Function to fetch data, train the model, and predict a price for a given date
def predict_price_for_date(result_label):
    try:
        # Fetch data
        btc_data = yf.download('BTC-USD', start='2015-01-01', end=datetime.today().strftime('%Y-%m-%d'))
        btc_data = btc_data[['Close']].dropna()

        # Preprocess data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(btc_data)

        # Prepare dataset for LSTM
        def create_dataset(data, time_step):
            X = []
            for i in range(len(data) - time_step):
                X.append(data[i:i + time_step, 0])
            return np.array(X)

        time_step = 60
        X = create_dataset(scaled_data, time_step)
        y = scaled_data[time_step:]

        # Convert data to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)

        # Reshape the data to have the shape [batch_size, seq_len, input_size]
        # Here, input_size is 1 (since we're using only 'Close' price)
        X = X.view(X.shape[0], X.shape[1], 1)  # Reshaping to [batch_size, seq_len, input_size]

        # Hyperparameters
        input_size = 1  # Number of features (close price)
        hidden_size = 50  # Number of LSTM units in the hidden layer
        output_size = 1  # We're predicting a single value (next day's close price)
        num_layers = 2  # Number of LSTM layers
        num_epochs = 5
        batch_size = 32
        learning_rate = 0.001

        # Initialize the model
        model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(num_epochs):
            for i in range(0, len(X) - batch_size, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Get the user-entered date
        future_date_str = date_entry.get()
        future_date = datetime.strptime(future_date_str, "%Y-%m-%d")
        days_to_predict = (future_date - datetime.today()).days

        if days_to_predict <= 0:
            messagebox.showerror("Error", "Enter a future date.")
            return

        # Predict future prices
        last_60_days = torch.tensor(scaled_data[-time_step:], dtype=torch.float32).to(device)
        last_60_days = last_60_days.view(1, time_step, 1)  # Reshape to [1, time_step, input_size]

        for _ in range(days_to_predict):
            predicted_price_scaled = model(last_60_days)
            predicted_price = predicted_price_scaled.item()
            last_60_days = torch.cat((last_60_days[:, 1:, :], torch.tensor([[[predicted_price]]], dtype=torch.float32).to(device)), dim=1)

        # Rescale the final predicted price
        final_price = scaler.inverse_transform([[predicted_price]])
        result_label.config(text=f"Predicted Bitcoin Price on {future_date_str}: ${final_price[0][0]:.2f}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")

# Build the GUI of predictor
def prediction():
    pg = tkinter.Toplevel(home)
    pg.title("Bitcoin Price Predictor")
    pg.geometry("500x300")
    pg.configure(bg="#001000")
# Add widgets
    title_label = tkinter.Label(pg, text="Bitcoin Price Predictor", font=("Helvetica", 16))
    title_label.pack(pady=60)

    date_label = tkinter.Label(pg, text="Enter Future Date (YYYY-MM-DD):", font=("Helvetica", 12))
    date_label.pack(pady=5)
    global date_entry
    date_entry = ttk.Entry(pg, font=("Helvetica", 12))
    date_entry.pack(pady=5)

    predict_button = ttk.Button(pg, text="Predict Price", command=lambda: predict_price_for_date(result_label))
    predict_button.pack(pady=30)
    
    # Create a frame for the result label and define it properly
    result_frame = tkinter.Frame(pg, borderwidth=2, relief="solid", padx=10, pady=10)
    result_frame.pack(pady=30)

# Define result_label inside this frame
    result_label = tkinter.Label(result_frame, text="", font=("Arial", 14), fg="green")
    result_label.pack()
    

    
   
    
# Function to fetch and display Bitcoin exchange rate graph
def bitcoingraph():
    # Replace 'YOUR_API_KEY' with your actual CoinAPI key
    api_key = 'EF212FDE-D84C-4093-B132-125C8E536F2C'
    headers = {'X-CoinAPI-Key': api_key}

# Define the currency and Bitcoin symbols
    currency = fiat_combobox.get()
    crypto = 'BTC'

# Define the time range for the data
    start_date = '2022-01-01'
    end_date = '2023-01-07'

# Fetch historical exchange rate data
    url = f'https://rest.coinapi.io/v1/exchangerate/{crypto}/{currency}/history?period_id=1DAY&time_start={start_date}T00:00:00&time_end={end_date}T00:00:00'
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
       data = response.json()
    # Extract dates and rates
       dates = [datetime.strptime(entry['time_period_start'][:10], '%Y-%m-%d') for entry in data]
       rates = [entry['rate_close'] for entry in data]
    # Convert datetime objects to numerical format
       date_nums = mdates.date2num(dates)
    # Generate a sequence of integers for the x-axis
       x_values = list(range(len(dates)))

    # Create a 3D plot
       fig = plt.figure(figsize=(12, 8))
       ax = fig.add_subplot(111, projection='3d')

    # Plot the data
       ax.plot(x_values, date_nums, rates, marker='o', linestyle='-', color='b')

    # Set labels
       ax.set_xlabel('Time (Days)')
       ax.set_ylabel('Date')
       ax.set_zlabel(f'Exchange Rate ({currency} per {crypto})')
       ax.set_title(f'{currency} to {crypto} Exchange Rate Over Time')

    # Format the y-axis to display dates
       ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Rotate date labels for better readability
       plt.setp(ax.get_yticklabels(), rotation=45, ha='right')

       plt.show()
    else:
       print(f'Error fetching data: {response.status_code} - {response.text}')
# Function to convert Bitcoin to fiat currency
def convert_crypto():
    try:
        crypto = crypto_combobox.get()
        fiat_currency = fiat_combobox.get()
        crypto_amount = float(crypto_entry.get())

        # Check if the selected crypto is BTC
        if crypto != "BTC":
            result_label.config(text="Error: Only BTC conversion is supported!")
            return

        # Perform conversion using live rates
        price = btc_converter.get_latest_price(fiat_currency)
        converted_amount = price * crypto_amount

        # Display the result
        result_label.config(text=f"{crypto_amount} {crypto} = {fiat_currency} {converted_amount:.2f}")
    except ValueError:
        result_label.config(text="Error: Please enter a valid amount!")
    except Exception as e:
        result_label.config(text=f"Error: {e}")

# Bitcoin converter interface
def bitcoin():
    imp = tkinter.Toplevel(home)
    imp.title("Crypto to Money Converter")
    imp.geometry("450x350")
    imp.configure(bg="#001000")
   

    # Title label
    tkinter.Label(imp, text="Crypto to Money Converter", font=("Arial", 18, "bold")).pack(pady=10)
    
    # Cryptocurrency selection
    tkinter.Label(imp, text="Select Cryptocurrency:").pack(pady=5)
    global crypto_combobox
    crypto_combobox = ttk.Combobox(imp, values=["BTC"])  # Only BTC supported
    crypto_combobox.pack()
    crypto_combobox.set("BTC")  # Default selection

    # Cryptocurrency amount input
    tkinter.Label(imp, text="Enter Cryptocurrency Amount:").pack(pady=5)
    global crypto_entry
    crypto_entry = tkinter.Entry(imp)
    crypto_entry.pack(pady=5)

    # Fiat currency selection
    tkinter.Label(imp, text="Select Fiat Currency:").pack(pady=5)
    global fiat_combobox
    fiat_combobox = ttk.Combobox(imp, values=["USD", "EUR", "GBP",  "AUD", "CAD", "JPY", "CHF"])
    fiat_combobox.pack()
    fiat_combobox.set("USD")  # Default selection

    # Convert button
    tkinter.Button(imp, text="Convert", command=convert_crypto ,activebackground="green",     # Background color when clicked
    activeforeground="yellow" ).pack(pady=20)

    # Frame for result
    result_frame = tkinter.Frame(imp, borderwidth=2, relief="solid", padx=10, pady=10)
    result_frame.pack(pady=10)
    global result_label
    result_label = tkinter.Label(result_frame, text="", font=("Arial", 14), fg="green")
    result_label.pack()
    
    # Button to open the Bitcoin graph
    tkinter.Button(imp, text="Graph", bg="pink", command=bitcoingraph ,activebackground="green",     # Background color when clicked
    activeforeground="yellow" ).pack(pady=30)

def get_exchange_rates(api_key, base_currency, target_currencies):
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base_currency}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data['result'] == 'success':
            rates = data['conversion_rates']
            # Filter only the target currencies
            filtered_rates = {currency: rates.get(currency) for currency in target_currencies}
            return filtered_rates
        else:
            print("Error: Invalid response from the API.")
            return {}
    else:
        print(f"Error: Failed to fetch data (Status Code: {response.status_code})")
        return {}
  

# Function to display a dummy currency graph
def graph():
 api_key = 'adccf6fba77803b80b484db2'  # Replace with your actual API key
 base_currency = 'USD'  # Change to any base currency you prefer
 target_currencies = ['EUR', 'GBP', 'INR', 'AUD', 'CAD', 'JPY']  # List of target currencies

# Fetch the exchange rates
 exchange_rates = get_exchange_rates(api_key, base_currency, target_currencies)

# Check if we received any valid exchange rates
 if exchange_rates:
    # Convert the exchange rates to a pandas DataFrame
    rates_df = pd.DataFrame.from_dict(exchange_rates, orient='index', columns=['Exchange Rate'])
    rates_df.index.name = 'Currency'

    # Reverse the exchange rates: stronger currencies have lower values for 1 USD
    rates_df['Strength'] = 1 / rates_df['Exchange Rate']  # Inverse to show strength

    # Sort the DataFrame by strength (highest strength at the top)
    rates_df = rates_df.sort_values('Strength', ascending=False)

    # 3D Bar Plot with animation
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data for 3D plot
    x = np.arange(len(rates_df))
    y = np.zeros_like(x)  # Base level for bars
    z = np.zeros_like(x)  # Z-axis (height)

    # Set the bar heights based on strength (higher for stronger currencies)
    dx = np.ones_like(x)  # Bar width in the x direction
    dy = np.ones_like(x)  # Bar depth in the y direction
    dz = rates_df['Strength'].values  # Bar heights based on strength

    # Bar colors (optional, customize as needed)
    colors = plt.cm.viridis(np.linspace(0, 1, len(rates_df)))

    # Create 3D bars
    ax.bar3d(x, y, z, dx, dy, dz, color=colors)

    # Set axis labels and title
    ax.set_xlabel('Currencies')
    ax.set_ylabel('Y')
    ax.set_zlabel('Strength')
    ax.set_title(f'3D Currency Strength Comparison for 1 {base_currency}')

    # Set the x-ticks to currency names
    ax.set_xticks(x)
    ax.set_xticklabels(rates_df.index, rotation=45)

    # Add simple animation to rotate the 3D plot
    def update_rotation(i):
        ax.view_init(elev=20, azim=i)

    # Animation loop
    for i in range(360):
        update_rotation(i)
        plt.pause(0.01)

    plt.show()

 else:
    print("No valid exchange rates were fetched.")

  
   
# Function to convert fiat currencies
def conversion():
    url = "https://currency-converter18.p.rapidapi.com/api/v1/convert"

    currency_1 = combobox1.get()
    currency_2 = combobox2.get()
    amount = value.get()

    querystring = {"from":currency_1,"to":currency_2,"amount":amount}

    if currency_2 == 'USD':
        symbol = '$'
    elif currency_2 == 'INR':
        symbol = '₹'
    elif currency_2 == 'EUR':
        symbol = '€'
    elif currency_2 == 'BRL':
        symbol = 'R$'
    elif currency_2 == 'CAD':
        symbol = 'CA $'
        

    headers = {
        'x-rapidapi-host': "currency-converter18.p.rapidapi.com",
        'x-rapidapi-key': "90c59d6c9fmsh4599f814e2ffc92p17fc6djsndeaa0265ac61"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)

    data = json.loads(response.text)
    converted_amount = data["result"]["convertedAmount"]
    formatted = symbol + " {:,.2f}".format(converted_amount)

    label['text'] = formatted
    
    print(converted_amount, formatted)

# Currency converter interface
def currency():
    root = tkinter.Toplevel(home)
    root.title("Currency Converter")
    root.geometry("300x400")
    root.configure(bg="#001000")
    # image1=Image.open(r"/Users/srishtigupta/Desktop/FYsempythonpro/moneyconvert.jpeg")
    # image1=image1.resize((700,450))
    # image1_convert=ImageTk.PhotoImage(image1)
    # image1_label=tkinter.Label(root,image=image1_convert)
    # image1_label.pack()
    tkinter.Label(root, text=" Currency Converter", font=("Arial", 18, "bold")).pack(pady=10)
    
    global combobox1, combobox2, value, label
    tkinter.Label(root, text="From Currency:").pack()
    combobox1 = ttk.Combobox(root, values=["USD", "EUR", "GBP", "INR", "AUD", "CAD", "JPY", "CHF"])
    combobox1.pack()
    tkinter.Label(root, text="To Currency:").pack()
    combobox2 = ttk.Combobox(root, values=["USD", "EUR", "GBP", "INR", "AUD", "CAD", "JPY", "CHF"])
    combobox2.pack()
    value = tkinter.Entry(root)
    value.pack()
    label = tkinter.Label(root, text="")
    label.pack(pady=10)
    tkinter.Button(root, text="Convert", bg="pink", command=conversion ,activebackground="green",     # Background color when clicked
    activeforeground="yellow" ).pack(pady=30)
    tkinter.Button(root, text="Graph", bg="pink", command=graph ,activebackground="green",     # Background color when clicked
    activeforeground="yellow" ).pack(pady=30)

# Main application window
home = tkinter.Tk()
home.title("Home")
home.geometry("600x600")
home.configure(bg="#000822")

# Background image
Imagepath = "/Users/srishtigupta/Desktop/FYsempythonpro/testing/horizontal_curbackground.jpeg" 
image_open = Image.open(Imagepath)
image_open=image_open.resize((700,450))
Image_convert = ImageTk.PhotoImage(image_open)
tkinter.Label(home, image=Image_convert).pack()
# Buttons to navigate
tkinter.Button(home, text="CURRENCY CONVERTER", bg="pink", command=currency, activebackground="green",     # Background color when clicked
    activeforeground="yellow" ).pack(pady=50)
tkinter.Button(home, text="BITCOIN CONVERTER", bg="pink", command=bitcoin ,activebackground="green",     # Background color when clicked
    activeforeground="yellow" ).pack(pady=50)
tkinter.Button(home, text="PREDICTION", bg="pink", command=prediction ,activebackground="green",     # Background color when clicked
    activeforeground="yellow" ).pack(pady=50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Start Tkinter loop
home.mainloop()