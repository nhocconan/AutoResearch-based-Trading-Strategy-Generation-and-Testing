#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Chandelier Exit trend with 1w trend filter and volume confirmation
# Uses Chandelier Exit (ATR-based trailing stop) to ride trends with dynamic exits.
# Filters by weekly EMA50 trend direction and volume spike (2x 20-period EMA).
# Designed for low-frequency trades (target 30-100 total) to minimize fee drag and work in both bull/bear markets.

name = "1d_ChandelierExit_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate ATR(22) for Chandelier Exit
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=22, adjust=False, min_periods=22).mean().values
    
    # Chandelier Exit for long and short
    # Long exit: highest high since entry minus ATR*3
    # Short exit: lowest low since entry plus ATR*3
    # We'll track highest high and lowest low since position entry
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    # Volume spike (2x 20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 and ATR have enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Initialize tracking for new position
            highest_since_entry[i] = high[i]
            lowest_since_entry[i] = low[i]
            
            # Enter long: price above 1w EMA50 with volume spike
            if close[i] > ema50_1w_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price below 1w EMA50 with volume spike
            elif close[i] < ema50_1w_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            lowest_since_entry[i] = lowest_since_entry[i-1]  # carry forward
            
            # Chandelier Exit long: price drops below highest high - 3*ATR
            chandelier_long = highest_since_entry[i] - 3.0 * atr[i]
            if close[i] < chandelier_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            highest_since_entry[i] = highest_since_entry[i-1]  # carry forward
            
            # Chandelier Exit short: price rises above lowest low + 3*ATR
            chandelier_short = lowest_since_entry[i] + 3.0 * atr[i]
            if close[i] > chandelier_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals