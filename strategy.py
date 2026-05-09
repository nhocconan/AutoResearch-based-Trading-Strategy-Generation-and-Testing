#!/usr/bin/env python3
name = "6H_4WkHigh_Low_MeanReversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 4-week high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 20-day high and low (4 weeks)
    high_20d = pd.Series(close_1d).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(close_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 6h timeframe
    high_20d_aligned = align_htf_to_ltf(prices, df_1d, high_20d)
    low_20d_aligned = align_htf_to_ltf(prices, df_1d, low_20d)
    
    # Calculate 6-period RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if 4-week data not ready
        if np.isnan(high_20d_aligned[i]) or np.isnan(low_20d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Mean reversion: long near 4-week low, short near 4-week high
        range_width = high_20d_aligned[i] - low_20d_aligned[i]
        if range_width < 1e-10:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Position within the 4-week range (0 = at low, 1 = at high)
        position_in_range = (close[i] - low_20d_aligned[i]) / range_width
        
        if position == 0:
            # Enter long near bottom of range (oversold) with RSI confirmation
            if position_in_range < 0.2 and rsi[i] < 35:
                signals[i] = 0.25
                position = 1
            # Enter short near top of range (overbought) with RSI confirmation
            elif position_in_range > 0.8 and rsi[i] > 65:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to middle of range or overbought
            if position_in_range > 0.5 or rsi[i] > 65:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to middle of range or oversold
            if position_in_range < 0.5 or rsi[i] < 35:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals