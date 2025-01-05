from tkinter import *
from tkinter import ttk
from forex_python.bitcoin import BtcConverter

# Initialize the BtcConverter
btc_converter = BtcConverter()

# Function to convert cryptocurrency to fiat money
def convert_crypto():
    try:
        crypto = crypto_combobox.get()
        fiat_currency = fiat_combobox.get()
        crypto_amount = float(crypto_entry.get())

        # Only supports BTC conversion, so crypto must be BTC
        if crypto != "BTC":
            result_label.config(text="Error: Only BTC conversion is supported!")
            return

        # Get the latest price and perform the conversion
        price = btc_converter.get_latest_price(fiat_currency)
        converted_amount = price * crypto_amount

        # Display the result
        result_label.config(text=f"{crypto_amount} {crypto} = {fiat_currency} {converted_amount:.2f}")
    except ValueError:
        result_label.config(text="Error: Please enter a valid amount!")
    except Exception as e:
        result_label.config(text=f"Error: {e}")

# Create the main application window
root = Tk()
root.title("Crypto to Money Converter")
root.geometry("450x350")

# Title label
title_label = Label(root, text="Crypto to Money Converter", font=("Arial", 18, "bold"))
title_label.pack(pady=10)

# Input for cryptocurrency type
crypto_label = Label(root, text="Select Cryptocurrency:")
crypto_label.pack(pady=5)
crypto_combobox = ttk.Combobox(root, values=["BTC"])  # Only BTC supported
crypto_combobox.pack()
crypto_combobox.set("BTC")  # Default value

# Input for cryptocurrency amount
crypto_amount_label = Label(root, text="Enter Cryptocurrency Amount:")
crypto_amount_label.pack(pady=5)
crypto_entry = Entry(root)
crypto_entry.pack(pady=5)   

# Input for fiat currency type
fiat_label = Label(root, text="Select Fiat Currency:")
fiat_label.pack(pady=5)
fiat_combobox = ttk.Combobox(root, values=["USD", "EUR", "GBP", "INR", "AUD", "CAD", "JPY", "CHF"])
fiat_combobox.pack()
fiat_combobox.set("USD")  # Default value

# Convert button
convert_button = Button(root, text="Convert", command=convert_crypto)
convert_button.pack(pady=20)

# Frame for result
result_frame = Frame(root, borderwidth=2, relief="solid", padx=10, pady=10)
result_frame.pack(pady=10)

# Result label inside the frame
result_label = Label(result_frame, text="", font=("Arial", 14), fg="green")
result_label.pack()

# Run the main loop
root.mainloop()