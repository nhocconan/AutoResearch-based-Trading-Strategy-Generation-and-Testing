#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "Vinz Win BTC Strategy Auto 1m"
timeframe = "1m"
leverage = 1

# Strategy Parameters
SESSION_START_HOUR = 0
SESSION_END_HOUR = 9
SL_BUFFER_PRICE = 5.0  # Approximate based on BTC tick value
RR_MULT = 2.0
MIN_BODY_RATIO = 0.4

def generate_signals(prices):
    # Ensure prices is a DataFrame
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("Prices must be a pandas DataFrame")
    
    n = len(prices)
    if n == 0:
        return np.array([], dtype=int)
    
    # Extract numpy arrays for speed and safety
    required_cols = ['open_time', 'open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in prices.columns:
            raise ValueError(f"Missing column: {col}")
            
    opens = prices['open'].values
    highs = prices['high'].values
    lows = prices['low'].values
    closes = prices['close'].values
    open_times = pd.to_datetime(prices['open_time'], utc=True)
    
    signals = np.zeros(n, dtype=int)
    
    # State variables
    in_position = False
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    
    # Loop through bars
    for i in range(n):
        # Set signal for current bar based on state from previous bar
        signals[i] = 1 if in_position else 0
        
        # Get current bar data
        o = opens[i]
        h = highs[i]
        l = lows[i]
        c = closes[i]
        t = open_times.iloc[i]

        # Repo data stores timestamps as datetime-like values, not raw epoch ms.
        utc_hour = int(t.hour)
        in_session = (SESSION_START_HOUR <= utc_hour < SESSION_END_HOUR)
        
        # Exit Logic (Check if SL/TP hit during current bar)
        if in_position:
            # If low touches SL or high touches TP, exit for next bar
            if l <= sl_price or h >= tp_price:
                in_position = False
                entry_price = 0.0
                sl_price = 0.0
                tp_price = 0.0
        
        # Entry Logic (Only if not in position)
        else:
            if i >= 2 and in_session:
                # Pattern: 2 Red followed by 1 Green
                is_green = c > o
                is_red_1 = closes[i-1] < opens[i-1]
                is_red_2 = closes[i-2] < opens[i-2]
                
                # Body Ratio Filter on Current Candle
                range_size = h - l
                body_size = abs(c - o)
                body_ratio = (body_size / range_size) if range_size > 0 else 0.0
                
                if is_green and is_red_1 and is_red_2 and (body_ratio >= MIN_BODY_RATIO):
                    # Enter Long for next bar
                    in_position = True
                    entry_price = c
                    sl_price = l - SL_BUFFER_PRICE
                    risk = entry_price - sl_price
                    tp_price = entry_price + (risk * RR_MULT)
    
    return signals
