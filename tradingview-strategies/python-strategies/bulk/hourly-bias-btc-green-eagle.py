#!/usr/bin/env python3
import pandas as pd
import numpy as np

name = "Hourly Bias on BTC in Bullish USA Session Green Eagle"
timeframe = "1h"
leverage = 1

def generate_signals(prices):
    n = len(prices)
    if n == 0:
        return np.array([], dtype=int)
    
    signals = np.zeros(n, dtype=int)
    
    # Repo data already stores UTC-compatible datetimes; do not force ms coercion.
    dt = pd.to_datetime(prices['open_time'], utc=True)
    hours = dt.dt.hour.values
    weekdays = dt.dt.weekday.values
    
    # Calculate True Range
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR
    tr_series = pd.Series(tr)
    atr2 = tr_series.rolling(window=2).mean().values
    atr14 = tr_series.rolling(window=14).mean().values
    
    # Filters
    # Pine dayofweek: Mon=1, Tue=2, Thu=4, Sat=6
    # Python weekday: Mon=0, Tue=1, Thu=3, Sat=5
    day_filter = np.isin(weekdays, [0, 1, 3, 5])
    vol_filter = (atr2 <= atr14) & (~np.isnan(atr2)) & (~np.isnan(atr14))
    
    # Conditions based on previous bar to avoid lookahead
    # Entry: Hour 14
    entry_cond = (hours == 14) & day_filter & vol_filter
    # Exit: Hour 0
    exit_cond = (hours == 0)
    
    position = 0
    for i in range(1, n):
        # Evaluate conditions at i-1 to determine position for i
        if entry_cond[i-1] and position == 0:
            position = 1
        elif exit_cond[i-1] and position == 1:
            position = 0
        signals[i] = position
        
    return signals
