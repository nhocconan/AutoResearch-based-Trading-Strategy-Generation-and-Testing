#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime

name = "BTC Volatility Band Strategy"
timeframe = "1d"
leverage = 1

def generate_signals(prices):
    """
    Generates trading signals based on BTC Volatility Band Strategy logic.
    Returns a numpy array of integers (1=Long, -1=Short, 0=Flat) with length equal to input prices.
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("Input prices must be a pandas DataFrame")
    
    df = prices.copy()
    n = len(df)
    if n == 0:
        return np.array([], dtype=int)
    
    signals = np.zeros(n, dtype=int)
    
    # Extract columns
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # Handle open_time for date filtering - convert to timestamp floats
    if 'open_time' in df.columns:
        open_time_vals = df['open_time'].values
        # Convert datetime64 to timestamp floats if needed
        if open_time_vals.dtype == 'datetime64[ns]' or open_time_vals.dtype == 'datetime64[ms]':
            open_time_ts = open_time_vals.astype('datetime64[ms]').astype(np.float64)
        else:
            # Assume already timestamp in ms
            open_time_ts = open_time_vals.astype(np.float64)
    else:
        # No time column, skip date filter
        open_time_ts = np.zeros(n)
    
    # --- Indicators ---
    # CandleChange = ((close - close[1])/close)*100
    df['cc'] = ((df['close'] - df['close'].shift(1)) / df['close']) * 100.0
    
    ma_len = 7
    df['ma_out'] = df['cc'].rolling(window=ma_len).mean()
    df['std'] = df['cc'].rolling(window=ma_len).std()
    
    inner_band = 1.0
    outer_band = 2.0
    
    df['dev_inner'] = inner_band * df['std']
    df['dev_outer'] = outer_band * df['std']
    
    df['upper1'] = df['ma_out'] + df['dev_inner']
    df['lower1'] = df['ma_out'] - df['dev_inner']
    df['upper2'] = df['ma_out'] + df['dev_outer']
    df['lower2'] = df['ma_out'] - df['dev_outer']
    
    # --- Filters ---
    # SMAFilter: close[1] > sma(close[1], 50)
    df['sma50'] = df['close'].shift(1).rolling(window=50).mean()
    df['sma_filter_l'] = df['close'].shift(1) > df['sma50']
    df['sma_filter_s'] = df['close'].shift(1) < df['sma50']
    
    # PriceFilter: close > lowest(close, 7)
    df['lowest7'] = df['close'].rolling(window=7).min()
    df['price_filter_l'] = df['close'] > df['lowest7']
    df['highest7'] = df['close'].rolling(window=7).max()
    df['price_filter_s'] = df['close'] < df['highest7']
    
    # VolFilter
    df['vol_filter_l'] = (df['cc'] <= df['lower1']) & (df['cc'] > df['lower2'])
    df['vol_filter_s'] = (df['cc'] >= df['upper1']) & (df['cc'] < df['upper2'])
    
    df['long_filter'] = df['vol_filter_l'] & df['sma_filter_l'] & df['price_filter_l']
    df['short_filter'] = df['vol_filter_s'] & df['sma_filter_s'] & df['price_filter_s']
    
    # --- Risk & Exit Levels ---
    # Risk = (high[7] - low[7]) / 7
    df['risk'] = (df['high'].shift(7) - df['low'].shift(7)) / 7.0
    df['profit_dist'] = df['risk'] * 1.15
    df['loss_dist'] = df['risk'] * 0.65
    
    # --- Date Filter ---
    # Use timestamp comparison with matching types
    start_ts = 946684800000.0  # 2000-01-01 00:00:00 UTC in ms
    end_ts = 4102444800000.0  # 2100-01-01 00:00:00 UTC in ms
    time_condition = (open_time_ts >= start_ts) & (open_time_ts <= end_ts)
    
    # --- Stateful Loop for Signal Generation ---
    # To avoid lookahead, entry signals are based on previous bar (i-1)
    # Exit signals are based on current bar (i) intrabar high/low
    position = 0  # 0=Flat, 1=Long, -1=Short
    entry_price = 0.0
    stop_price = 0.0
    limit_price = 0.0
    
    # Pre-extract arrays for speed
    arr_high = df['high'].values
    arr_low = df['low'].values
    arr_close = df['close'].values
    arr_long_filter = df['long_filter'].values
    arr_short_filter = df['short_filter'].values
    arr_loss_dist = df['loss_dist'].values
    arr_profit_dist = df['profit_dist'].values
    
    for i in range(n):
        # Default signal is previous position (will be updated)
        signals[i] = 0
        
        # Check time condition
        if not time_condition[i]:
            position = 0
            entry_price = 0.0
            continue
        
        # 1. Check Exits for existing position using current bar High/Low
        if position != 0:
            # Update stops dynamically based on current bar's risk calculation
            loss_d = arr_loss_dist[i]
            profit_d = arr_profit_dist[i]
            
            if np.isnan(loss_d): loss_d = 0.0
            if np.isnan(profit_d): profit_d = 0.0
            
            if position == 1:
                stop_price = entry_price - loss_d
                limit_price = entry_price + profit_d
                # Check Stop/Limit hit
                if arr_low[i] <= stop_price or arr_high[i] >= limit_price:
                    position = 0
                    entry_price = 0.0
            elif position == -1:
                stop_price = entry_price + loss_d
                limit_price = entry_price - profit_d
                # Check Stop/Limit hit
                if arr_high[i] >= stop_price or arr_low[i] <= limit_price:
                    position = 0
                    entry_price = 0.0
        
        # 2. Check Entries if Flat (using previous bar's filters to avoid lookahead)
        if position == 0:
            if i > 0:
                lf = arr_long_filter[i-1]
                sf = arr_short_filter[i-1]
                
                if lf and not np.isnan(lf):
                    position = 1
                    entry_price = arr_close[i-1]
                    # Set initial stops based on entry bar risk
                    loss_d = arr_loss_dist[i-1]
                    profit_d = arr_profit_dist[i-1]
                    if np.isnan(loss_d): loss_d = 0.0
                    if np.isnan(profit_d): profit_d = 0.0
                    stop_price = entry_price - loss_d
                    limit_price = entry_price + profit_d
                elif sf and not np.isnan(sf):
                    position = -1
                    entry_price = arr_close[i-1]
                    loss_d = arr_loss_dist[i-1]
                    profit_d = arr_profit_dist[i-1]
                    if np.isnan(loss_d): loss_d = 0.0
                    if np.isnan(profit_d): profit_d = 0.0
                    stop_price = entry_price + loss_d
                    limit_price = entry_price - profit_d
        
        signals[i] = position
    
    return signals
