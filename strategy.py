#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6d_ema_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA10
    close_1d = df_1d['close'].values
    ema10_1d = pd.Series(close_1d).ewm(span=10, adjust=False).mean().values
    
    # Align EMA10 to 6h timeframe
    ema10_6h = align_htf_to_ltf(prices, df_1d, ema10_1d)
    
    # Volume confirmation (20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema10_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        # Exit conditions: price crosses EMA10 or volume fade
        exit_long = (close[i] <= ema10_6h[i]) or (not vol_confirm)
        exit_short = (close[i] >= ema10_6h[i]) or (not vol_confirm)
        
        if position == 1:  # Long position
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: Price above EMA10 + volume confirmation
            if close[i] > ema10_6h[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: Price below EMA10 + volume confirmation
            elif close[i] < ema10_6h[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals