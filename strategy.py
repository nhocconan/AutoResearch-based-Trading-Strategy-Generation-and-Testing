#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ChandelierExit_12hTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA20 for trend filter
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Calculate ATR for Chandelier exit
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[0], tr])
    atr = np.zeros(n)
    for i in range(n):
        if i < 21:
            atr[i] = np.nan
        else:
            atr[i] = np.mean(tr[i-20:i+1])
    
    # Calculate Chandelier Exit levels
    chandelier_long = np.zeros(n)
    chandelier_short = np.zeros(n)
    
    # Initialize with first values
    highest_high = high[0]
    lowest_low = low[0]
    
    for i in range(n):
        highest_high = max(highest_high, high[i])
        lowest_low = min(lowest_low, low[i])
        if not np.isnan(atr[i]):
            chandelier_long[i] = highest_high - 3.0 * atr[i]
            chandelier_short[i] = lowest_low + 3.0 * atr[i]
        else:
            chandelier_long[i] = np.nan
            chandelier_short[i] = np.nan
    
    # Volume confirmation (20-period average)
    vol_avg_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema20_12h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirmed = volume[i] > 1.2 * vol_avg_20[i]
        price = close[i]
        
        if position == 0:
            # Enter long: price above Chandelier long exit + trend up + volume
            if price > chandelier_long[i] and ema20_12h_aligned[i] > price and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price below Chandelier short exit + trend down + volume
            elif price < chandelier_short[i] and ema20_12h_aligned[i] < price and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below Chandelier long exit
            if price < chandelier_long[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above Chandelier short exit
            if price > chandelier_short[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals