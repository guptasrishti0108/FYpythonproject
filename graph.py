# Replace 'YOUR_API_KEY' with your actual CoinAPI key
api_key = 'EF212FDE-D84C-4093-B132-125C8E536F2C'
headers = {'X-CoinAPI-Key': api_key}

# Define the currency and Bitcoin symbols
currency = 'USD'
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