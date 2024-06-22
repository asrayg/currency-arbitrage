# import requests
# import math
# import logging
# from collections import defaultdict
# from tqdm import tqdm
# from decimal import Decimal, getcontext
# from multiprocessing import Pool, cpu_count
# import time

# # Set precision for Decimal calculations
# getcontext().prec = 50

# # Logging configuration
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Replace 'YOUR_API_KEY' with your actual API key
# API_KEY = '421987be0bd65f2be281c3e2'
# BASE_URL = 'https://v6.exchangerate-api.com/v6/'

# # Transaction fee as a percentage (e.g., 0 for no fees, 0.5 for 0.5% fees)
# TRANSACTION_FEE_PERCENT = Decimal(0)

# def get_exchange_rates(api_key):
#     try:
#         url = f'{BASE_URL}{api_key}/latest/USD'
#         response = requests.get(url)
#         response.raise_for_status()
#         data = response.json()
#         if 'conversion_rates' in data:
#             return data['conversion_rates']
#         else:
#             logging.error("Conversion rates not found in response")
#             return None
#     except requests.exceptions.RequestException as e:
#         logging.error(f"Error fetching exchange rates: {e}")
#         return None

# def structure_exchange_rates(rates):
#     structured_data = {}
#     currencies = list(rates.keys())
    
#     for i in range(len(currencies)):
#         for j in range(len(currencies)):
#             if i != j:
#                 curr1 = currencies[i]
#                 curr2 = currencies[j]
#                 rate = Decimal(rates[curr2]) / Decimal(rates[curr1])
#                 structured_data[(curr1, curr2)] = rate
    
#     return structured_data

# def apply_transaction_fee(rate):
#     fee_multiplier = Decimal(1) - (TRANSACTION_FEE_PERCENT / Decimal(100))
#     return rate * fee_multiplier

# def create_graph(data):
#     graph = defaultdict(list)
    
#     for (curr1, curr2), rate in data.items():
#         rate_with_fee = apply_transaction_fee(rate)
#         graph[curr1].append((curr2, -math.log(rate_with_fee)))
    
#     return graph

# def initialize(graph, source):
#     d = {node: float('Inf') for node in graph}
#     d[source] = 0
#     p = {node: None for node in graph}
#     return d, p

# def relax(node, neighbor, weight, d, p):
#     if d[neighbor] > d[node] + weight:
#         d[neighbor] = d[node] + weight
#         p[neighbor] = node

# def bellman_ford(args):
#     graph, source = args
#     d, p = initialize(graph, source)
#     nodes = list(graph.keys())
#     num_nodes = len(nodes)
    
#     for i in range(num_nodes - 1):
#         for node in graph:
#             for neighbor, weight in graph[node]:
#                 relax(node, neighbor, weight, d, p)
    
#     for node in graph:
#         for neighbor, weight in graph[node]:
#             if d[neighbor] > d[node] + weight:
#                 return True, p, source
    
#     return False, p, source

# def find_arbitrage(graph):
#     cycles = []
#     nodes = list(graph.keys())
    
#     with Pool(cpu_count()) as pool:
#         results = pool.map(bellman_ford, [(graph, node) for node in nodes])
    
#     for has_cycle, predecessors, source in results:
#         if has_cycle:
#             for n in nodes:
#                 cycle = reconstruct_cycle(predecessors, source, n)
#                 if cycle and len(cycle) > 3 and cycle not in cycles:
#                     cycles.append(cycle)
    
#     return cycles

# def reconstruct_cycle(predecessors, start_node, end_node):
#     arbitrage_cycle = []
#     visited = set()
#     current_node = end_node
    
#     while current_node not in visited:
#         visited.add(current_node)
#         arbitrage_cycle.append(current_node)
#         current_node = predecessors[current_node] 
#         if current_node is None:
#             return []
    
#     arbitrage_cycle = arbitrage_cycle[arbitrage_cycle.index(current_node):]  
#     arbitrage_cycle.append(current_node)
#     return arbitrage_cycle

# def calculate_profit(cycle, rates):
#     profit = Decimal(1.0)
#     for i in range(len(cycle) - 1):
#         rate = apply_transaction_fee(rates[(cycle[i], cycle[i + 1])])
#         profit *= rate
#     profit_percentage = (profit - Decimal(1)) * Decimal(100)
#     return profit_percentage

# def write_graph_to_txt(graph, filename='exchange_rates_graph.txt'):
#     try:
#         with open(filename, 'w') as file:
#             for node in graph:
#                 for neighbor, weight in graph[node]:
#                     file.write(f"{node} -> {neighbor}: {math.exp(-weight)}\n")
#         logging.info(f"Graph written to {filename}")
#     except Exception as e:
#         logging.error(f"Error writing graph to file: {e}")

#     # rates_time = time.time()
#     # rates_end_time = time.time()
#     # logging.info(f"Time taken: {rates_time - rates_end_time:.2f} seconds")
# if __name__ == '__main__':
    
#     rates = get_exchange_rates(API_KEY)

#     if rates:
#         structured_data = structure_exchange_rates(rates)
        
#         graph = create_graph(structured_data)
        
#         write_graph_to_txt(graph)
#         start_time = time.time()

#         arbitrage_cycles = find_arbitrage(graph)
        
#         if arbitrage_cycles:
#             logging.info("Arbitrage detected")
#             for cycle in arbitrage_cycles:
#                 profit_percentage = calculate_profit(cycle, structured_data)
#                 logging.info(f"Cycle: {cycle} | Profit: {profit_percentage:.50f}%")
#         else:
#             logging.info("No arbitrage detected.")
#     else:
#         logging.error("Failed to fetch rates.")

#     end_time = time.time()
#     logging.info(f"Time taken: {end_time - start_time:.2f} seconds")
import asyncio
import aiohttp
import math
import logging
import time
from collections import defaultdict
from tqdm import tqdm
from decimal import Decimal, getcontext
from multiprocessing import Pool, cpu_count
import concurrent.futures

# Set precision for Decimal calculations
getcontext().prec = 50

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Replace 'YOUR_API_KEY' with your actual API key
API_KEY = '421987be0bd65f2be281c3e2'
BASE_URL = 'https://v6.exchangerate-api.com/v6/'

# Transaction fee as a percentage (e.g., 0 for no fees, 0.5 for 0.5% fees)
TRANSACTION_FEE_PERCENT = Decimal(0)

async def fetch_exchange_rates(api_key):
    async with aiohttp.ClientSession() as session:
        try:
            url = f'{BASE_URL}{api_key}/latest/USD'
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                if 'conversion_rates' in data:
                    return data['conversion_rates']
                else:
                    logging.error("Conversion rates not found in response")
                    return None
        except aiohttp.ClientError as e:
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
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(bellman_ford, (graph, node)) for node in nodes]
        for future in concurrent.futures.as_completed(futures):
            has_cycle, predecessors, source = future.result()
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

if __name__ == '__main__':
    start_time = time.time()

    rates = asyncio.run(fetch_exchange_rates(API_KEY))

    if rates:
        structured_data = structure_exchange_rates(rates)
        
        graph = create_graph(structured_data)
        
        write_graph_to_txt(graph)
        
        arbitrage_cycles = find_arbitrage(graph)
        
        if arbitrage_cycles:
            logging.info("Arbitrage detected")
            for cycle in arbitrage_cycles:
                profit_percentage = calculate_profit(cycle, structured_data)
                # Print all positive profits even if they are very small
                logging.info(f"Cycle: {cycle} | Profit: {profit_percentage:.50f}%")
        else:
            logging.info("No arbitrage detected.")
    else:
        logging.error("Failed to fetch rates.")

    end_time = time.time()
    logging.info(f"Time taken: {end_time - start_time:.2f} seconds")
