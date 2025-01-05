import tkinter
from PIL import Image , ImageTk
from tkinter import ttk
from forex_python.converter import CurrencyRates, CurrencyCodes
from forex_python.bitcoin import BtcConverter
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import requests
from datetime import datetime
import numpy as np
btc_converter = BtcConverter()
c = CurrencyRates()
codes = CurrencyCodes()

def bitcoingraph():
   bitgraph=tkinter.Toplevel(home)
    # Replace 'YOUR_API_KEY' with your actual CoinAPI key
   api_key = 'EF212FDE-D84C-4093-B132-125C8E536F2C'
   headers = {'X-CoinAPI-Key': api_key}

# Define the currency and Bitcoin symbols
   currency =fiat_combobox.get()
   crypto = 'BTC'

# Define the time range for the data
   start_date = '2023-01-01'
   end_date = '2023-01-07'

# Fetch historical exchange rate data
   url = f'https://rest.coinapi.io/v1/exchangerate/{crypto}/{currency}/history?period_id=1DAY&time_start={start_date}T00:00:00&time_end={end_date}T00:00:00'
   response =requests.get(url, headers=headers)


   if response.status_code == 200:
    data = response.json()
    # Extract dates and rates
    dates = [datetime.strptime(entry['time_period_start'][:10], '%Y-%m-%d') for entry in data]
    rates = [entry['rate_close'] for entry in data]

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(dates, rates, marker='o', linestyle='-', color='b')
    plt.xlabel('Date')
    plt.ylabel(f'Exchange Rate ({currency} per {crypto})')
    plt.title(f'{currency} to {crypto} Exchange Rate Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
   else:
    print(f'Error fetching data: {response.status_code} - {response.text}')
def convert_crypto():
    try:
        crypto = crypto_combobox.get()
        fiat_currency = fiat_combobox.get()
        crypto_amount = float(crypto_entry.get())

        # Only supports BTC conversion, so crypto must be BTC
        if crypto != "BTC":
            tkinter.result_label.config(text="Error: Only BTC conversion is supported!")
            return

        # Get the latest price and perform the conversion
        price = btc_converter.get_latest_price(fiat_currency)
        converted_amount = price * crypto_amount

        # Display the result
        tkinter.result_label.config(text=f"{crypto_amount} {crypto} = {fiat_currency} {converted_amount:.2f}")
    except ValueError:
        tkinter.result_label.config(text="Error: Please enter a valid amount!")
    except Exception as e:
        tkinter.result_label.config(text=f"Error: {e}")

def bitcoin():
   imp = tkinter.Toplevel(home)
   imp.title("Crypto to Money Converter")
   imp.geometry("450x350")

# Title label
   title_label = tkinter.Label(imp, text="Crypto to Money Converter", font=("Arial", 18, "bold"))
   title_label.pack(pady=10)

# Input for cryptocurrency type
   crypto_label = tkinter.Label(imp, text="Select Cryptocurrency:")
   crypto_label.pack(pady=5)
   crypto_combobox = ttk.Combobox(imp, values=["BTC"])  # Only BTC supported
   crypto_combobox.pack()
   crypto_combobox.set("BTC")  # Default value

# Input for cryptocurrency amount
   crypto_amount_label = tkinter.Label(imp, text="Enter Cryptocurrency Amount:")
   crypto_amount_label.pack(pady=5)
   crypto_entry = tkinter.Entry(imp)
   crypto_entry.pack(pady=5)

# Input for fiat currency type
   fiat_label = tkinter.Label(imp, text="Select Fiat Currency:")
   fiat_label.pack(pady=5)
   global fiat_combobox
   fiat_combobox = ttk.Combobox(imp, values=["USD", "EUR", "GBP", "INR", "AUD", "CAD", "JPY", "CHF"])
   fiat_combobox.pack()
   fiat_combobox.set("USD")  # Default value

# Convert button
   convert_button = tkinter.Button(imp, text="Convert", command=convert_crypto)
   convert_button.pack(pady=20)

# Frame for result
   result_frame = tkinter.Frame(imp, borderwidth=2, relief="solid", padx=10, pady=10)
   result_frame.pack(pady=10)

# Result label inside the frame
   result_label = tkinter.Label(result_frame, text="", font=("Arial", 14), fg="green")
   result_label.pack()
   button4=tkinter.Button(imp,text="graph",bg="pink",activebackground="green", activeforeground="yellow" , command=bitcoingraph)
   button4.pack(pady=30)

def graph():
   main = tkinter.Toplevel(home)
   days = np.arange(1, 11)  # Days from 1 to 10
   exchange_rate = [0.93, 0.94, 0.92, 0.91, 0.95, 0.96, 0.97, 0.95, 0.94, 0.93]  # Example rates

# Plotting the data
   plt.figure(figsize=(10, 6))
   plt.plot(days, exchange_rate, marker='o', color='blue', label='USD to EUR')

# Adding labels, title, and grid
   plt.title('Currency Conversion: USD to EUR', fontsize=16)
   plt.xlabel('Day', fontsize=12)
   plt.ylabel('Exchange Rate (EUR per USD)', fontsize=12)
   plt.grid(True, linestyle='--', alpha=0.7)
   plt.legend()
   plt.tight_layout()

# Show the graph
   plt.show()


def conversion():
   from_currency =combobox1.get()
   to_currency = combobox2.get()
   amount = value.get()
   converted_amount = c.convert(from_currency, to_currency, amount)
   symbol = codes.get_symbol(to_currency)
   label.config(text=f"Converted Amount: {symbol} {converted_amount}")
   label.pack()

def currency():
  root=tkinter.Toplevel(home)
  root.title("Currency Converter")
  root.config(bg="light blue")
  root.geometry("300x400")
  imagepath="/Users/srishtigupta/Desktop/pythonproject/currencypic.jpg"
  image=Image.open(imagepath)
  image_convert=ImageTk.PhotoImage(image)
  image_label=tkinter.Label(root,image=image_convert)
  image_label.pack(pady=10)
  list1=["INR","USD"]
  label1=tkinter.Label(text="from currency")
  label1.pack()
  combobox1=ttk.Combobox(root,value=list1)
  combobox1.pack()
  list2=["INR","USD"]
  label2=tkinter.Label(text="to currency")
  label2.pack()
  combobox2=ttk.Combobox(root,value=list2)
  combobox2.pack()
  value=tkinter.Entry(root)
  value.pack()
  label=tkinter.Label(root,text="")
  label.pack(pady=10)
  button=tkinter.Button(root,text="convert",bg="pink",command=conversion)
  button.pack(pady=30)
  button2=tkinter.Button(root,text="graph",bg="pink",activebackground="green", activeforeground="yellow" , command=graph)
  button2.pack(pady=30)
 

# Create the main window
home= tkinter.Tk()
home.title("Home")
home.geometry("600x600")
Imagepath="/Users/srishtigupta/Desktop/pythonproject/curbackground.jpeg"
image_open=Image.open(Imagepath)
Image_convert=ImageTk.PhotoImage(image_open)
Image_label=tkinter.Label(home,image=Image_convert)
Image_label.place(relheight=1,relwidth=1)

bitbutton=tkinter.Button(home,text="BITCOIN CONVERTER",bg="pink",activebackground="green", activeforeground="yellow" , command=bitcoin)
bitbutton.pack(pady=200)
currencybutton=tkinter.Button(home,text="CURRENCY CONVERTER",bg="pink",activebackground="green", activeforeground="yellow" , command=currency)
currencybutton.pack(pady=200)

# Start the Tkinter event loop
home.mainloop()