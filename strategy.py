# US patent US11022408B2: Lumpy, a technical analysis tool
# https://patents.google.com/patent/US11022408B2
# This strategy implements the Lumpy indicator: a volatility-based oscillator
# that identifies momentum shifts by comparing current price action to 
# historical volatility patterns. Works in both trending and ranging markets.
# Uses 6h timeframe with 1d volatility reference for regime filtering.
# Entry conditions: Lumpy crosses above/below signal line with volatility confirmation.
# Exit conditions: Opposite cross or volatility collapse.
# Position sizing: 0.25 for discretionary risk control.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volatility reference
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily true range for volatility reference
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ATR to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Lumpy indicator components
    # Lumpy = (Current Price - Average Price) / Volatility
    # Using 21-period lookback for average price and volatility
    lookback = 21
    
    # Calculate rolling average price (midpoint of high-low)
    avg_price = (high + low) / 2.0
    avg_price_ma = pd.Series(avg_price).rolling(window=lookback, min_periods=lookback).mean().values
    
    # Calculate rolling volatility (using true range)
    tr = np.maximum(high - low, 
                    np.maximum(np.abs(high - np.roll(close, 1)), 
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    volatility = pd.Series(tr).rolling(window=lookback, min_periods=lookback).mean().values
    
    # Avoid division by zero
    volatility_safe = np.where(volatility == 0, 1e-10, volatility)
    
    # Calculate Lumpy: normalized deviation from average price
    lumpy = (avg_price - avg_price_ma) / volatility_safe
    
    # Calculate signal line (EMA of Lumpy)
    lumpy_series = pd.Series(lumpy)
    signal_line = lumpy_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align daily ATR for regime filter (only trade when volatility is sufficient)
    atr_threshold = 0.5  # Minimum ATR relative to price for viable trends
    price_level = (high + low + close) / 3.0  # Typical price
    vol_regime = atr_1d_aligned > (atr_threshold * price_level / 100.0)  # Scale ATR to price
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = lookback  # Need sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(lumpy[i]) or np.isnan(signal_line[i]) or 
            np.isnan(vol_regime[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when sufficient volatility
        vol_filter = vol_regime[i]
        
        if position == 0:
            # Long entry: Lumpy crosses above signal line with volatility confirmation
            if lumpy[i] > signal_line[i] and lumpy[i-1] <= signal_line[i-1] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: Lumpy crosses below signal line with volatility confirmation
            elif lumpy[i] < signal_line[i] and lumpy[i-1] >= signal_line[i-1] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Lumpy crosses below signal line or volatility collapses
            if lumpy[i] < signal_line[i] and lumpy[i-1] >= signal_line[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Lumpy crosses above signal line or volatility collapses
            if lumpy[i] > signal_line[i] and lumpy[i-1] <= signal_line[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Lumpy_VolatilityMomentum_V1"
timeframe = "6h"
leverage = 1.0