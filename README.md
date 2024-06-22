# Currency Arbitrage Detection

This repository contains scripts for detecting arbitrage opportunities in currency exchange rates using various algorithms and optimizations. The scripts are organized into multiple files, each showcasing different approaches and optimizations. The project also includes a Flask server to run the detection script continuously and serve results via a web API. The system uses advanced algorithms, graph theory, and machine learning techniques to enhance performance and provide trading recommendations.

# Scientific Paper

Wrote a scientific paper on the subject matter. Check out the pdf in the repo.

## Project Structure

```
currency-arbitrage/
├── mode;
│   ├── Model.ipynb
│   └── modelApp.py
├── OtherMethods
│   ├── using-hardware.py
│   └── waytoolong.py
├── templates
│   └── index.html
├── .env
├── app.py
├── arbitrage.py
├── exchange_rates_graph.txt
├── exchange_rates.txt
├── README.md
└── requirements.txt
```

## Overview

- `app.py`: The Flask application that runs the arbitrage detection script and serves the results through a web API.
- `arbitrage.py`: Contains the core logic for fetching exchange rates, detecting arbitrage opportunities, and calculating profits.
- `OtherMethods/using-hardware.py`: An optimized version of the main script that utilizes hardware resources more efficiently.
- `OtherMethods/waytoolong.py`: A less optimized version of the script that takes longer to run.
- `templates/index.html`: A simple HTML template for the home page.
- `.env`: File containing environment variables such as the API key.
- `exchange_rates_graph.txt`: Contains the graph representation of the exchange rates with transaction fees applied.
- `exchange_rates.txt`: Contains raw exchange rate data fetched from the API.
- `requirements.txt`: Lists all the Python packages needed to run the Flask application.
- `README.md`: This README file.

## Setup Instructions

### Prerequisites

- Python 3.x
- Pip (Python package installer)

### Installation

1. **Clone the repository**:

   ```sh
   git clone https://github.com/yourusername/currency-arbitrage.git
   cd currency-arbitrage
   ```

2. **Create a virtual environment** (optional but recommended):

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

4. **Create a `.env` file** with your API key:

   ```env
   API_KEY=your_api_key_here
   ```

### Running the Application

1. **Run the Flask application**:

   ```sh
   python app.py
   ```

2. **Access the web interface**:

   Open your web browser and go to `http://127.0.0.1:5000/`.

3. **Detect arbitrage opportunities**:

   Click the "Detect Arbitrage" button on the home page to run the arbitrage detection script and display the results.

## Detailed Explanation

### Mathematical Foundations

Currency arbitrage involves exploiting differences in exchange rates across different markets to achieve profit. The goal is to find a sequence of currency exchanges that start and end with the same currency, resulting in a profit.

#### Bellman-Ford Algorithm

The Bellman-Ford algorithm is used to find the shortest paths from a single source vertex to all other vertices in a weighted graph. It is particularly useful for detecting negative-weight cycles, which correspond to arbitrage opportunities.

#### Floyd-Warshall Algorithm

The Floyd-Warshall algorithm is an all-pairs shortest path algorithm that can find shortest paths between all pairs of vertices in a weighted graph. This algorithm is used in `main.py` to optimize the detection of arbitrage opportunities.

### Optimizations

1. **Asynchronous Requests**: Using `aiohttp` for non-blocking HTTP requests.
2. **Parallel Processing**: Leveraging `multiprocessing` and `concurrent.futures` for parallel execution.
3. **Early Stopping**: Modifying the Bellman-Ford algorithm to stop early if no updates are made.

## Sentiment Analysis and Trading Recommendations
### Data Collection
News articles and social media posts related to various countries are collected using APIs like NewsAPI and Twitter API. The data is then preprocessed to remove noise and extract meaningful information.

### Sentiment Analysis
Sentiment analysis is performed on the collected data using the VADER sentiment analysis tool. This helps in determining the overall sentiment (positive, negative, neutral) towards a country's economy.

### Feature Extraction
Features such as sentiment score, GDP growth rate, and inflation rate are extracted from the data. These features are then used to train machine learning models to predict currency movements.

### Model Training
A Random Forest model is trained using the extracted features to predict the direction and magnitude of currency movements. The model is evaluated and fine-tuned to improve its accuracy.

### Trading Recommendations
Based on the model's predictions, a recommendation system provides trading advice. Traders can use this system to make informed decisions about buying, selling, or holding currency pairs.


## Scripts Overview

### `app.py`

```python
from flask import Flask, jsonify, render_template
from arbitrage import detect_arbitrage
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect-arbitrage')
def detect():
    results = detect_arbitrage()
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
```

### `arbitrage.py`

Contains the core logic for fetching exchange rates, detecting arbitrage opportunities, and calculating profits.

```python
import requests
import math
import logging
from collections import defaultdict
from decimal import Decimal, getcontext
from multiprocessing import Pool, cpu_count
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set precision for Decimal calculations
getcontext().prec = 50

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read API key from environment variable
API_KEY = os.getenv('API_KEY')
BASE_URL = 'https://v6.exchangerate-api.com/v6/'

# Transaction fee as a percentage (e.g., 0 for no fees, 0.5 for 0.5% fees)
TRANSACTION_FEE_PERCENT = Decimal(0)

def get_exchange_rates(api_key):
    try:
        url = f'{BASE_URL}{api_key}/latest/USD'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'conversion_rates' in data:
            return data['conversion_rates']
        else:
            logging.error("Conversion rates not found in response")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching exchange rates: {e}")
        return None

def structure_exchange_rates(rates):
    structured_data = {}
    currencies = list(rates.keys())
    
    for i in range(len(currencies)):
        for j in range(len(currencies)):
            if i != j:
                curr1 = currencies[i]
                curr2 = currencies[j]
                rate = Decimal(rates[curr2]) / Decimal(rates[curr1])
                structured_data[(curr1, curr2)] = rate
    
    return structured_data

def apply_transaction_fee(rate):
    fee_multiplier = Decimal(1) - (TRANSACTION_FEE_PERCENT / Decimal(100))
    return rate * fee_multiplier

def create_graph(data):
    graph = defaultdict(list)
    
    for (curr1, curr2), rate in data.items():
        rate_with_fee = apply_transaction_fee(rate)
        graph[curr1].append((curr2, -math.log(rate_with_fee)))
    
    return graph

def initialize(graph, source):
    d = {node: float('Inf') for node in graph}
    d[source] = 0
    p = {node: None for node in graph}
    return d, p

def relax(node, neighbor, weight, d, p):
    if d[neighbor] > d[node] + weight:
        d[neighbor] = d[node] + weight
        p[neighbor] = node

def bellman_ford(args):
    graph, source = args
    d, p = initialize(graph, source)
    nodes = list(graph.keys())
    num_nodes = len(nodes)
    
    for i in range(num_nodes - 1):
        for node in graph:
            for neighbor, weight in graph[node]:
                relax(node, neighbor, weight, d, p)
    
    for node in graph:
        for neighbor, weight in graph[node]:
            if d[neighbor] > d[node] + weight:
                return True, p, source
    
    return False, p, source

def find_arbitrage(graph):
    cycles = []
    nodes = list(graph.keys())
    
    with Pool(cpu_count()) as pool:
        results = pool.map(bellman_ford, [(graph, node) for node in nodes])
    
    for has_cycle, predecessors, source in results:
        if has_cycle:
            for n in nodes:
                cycle = reconstruct_cycle(predecessors, source, n)
                if cycle and len(cycle) > 3 and cycle not in cycles:
                    cycles.append(cycle)
    
    return cycles

def reconstruct_cycle(predecessors, start_node, end_node):
    arbitrage_cycle = []
    visited = set()
    current_node = end_node
    
    while current_node not in visited:
        visited.add(current_node)
        arbitrage_cycle.append(current_node)
        current_node = predecessors[current_node] 
        if current_node is None:
            return []
    
    arbitrage_cycle = arbitrage_cycle[arbitrage_cycle.index(current_node):]  
    arbitrage_cycle.append(current_node)
    return arbitrage_cycle

def calculate_profit(cycle, rates):
    profit = Decimal(1.0)
    for i in range(len(cycle) - 1):
        rate = apply_transaction_fee(rates[(cycle[i], cycle[i + 1])])
        profit *= rate
    profit_percentage = (profit - Decimal(1)) * Decimal(100)
    return profit_percentage

def write_graph_to_txt(graph, filename='exchange_rates_graph.txt'):
    try:
        with open(filename, 'w') as file:
            for node in graph:
                for neighbor, weight in graph[node]:
                    file.write(f"{node} -> {neighbor}: {math.exp(-weight)}\n")
        logging.info(f"Graph written to {filename}")
    except Exception as e:
        logging.error(f"Error writing graph to file: {e}")

def detect_arbitrage

():
    rates = get_exchange_rates(API_KEY)

    if rates:
        structured_data = structure_exchange_rates(rates)
        
        graph = create_graph(structured_data)
        
        write_graph_to_txt(graph)
        start_time = time.time()

        arbitrage_cycles = find_arbitrage(graph)
        
        results = []
        if arbitrage_cycles:
            logging.info("Arbitrage detected")
            for cycle in arbitrage_cycles:
                profit_percentage = calculate_profit(cycle, structured_data)
                results.append({
                    "cycle": cycle,
                    "profit": f"{profit_percentage:.50f}%"
                })
                logging.info(f"Cycle: {cycle} | Profit: {profit_percentage:.50f}%")
        else:
            logging.info("No arbitrage detected.")
        end_time = time.time()
        logging.info(f"Time taken: {end_time - start_time:.2f} seconds")
        return results
    else:
        logging.error("Failed to fetch rates.")
        return []
```

## Contributing

Feel free to contribute to this project by opening issues or submitting pull requests.

## License

This project is licensed under the MIT License.
