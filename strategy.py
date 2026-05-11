# 37
#!/usr/bin/env python3
name = "6h_WoW_Momentum_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for WoW momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Week-over-Week momentum (5-day momentum on daily)
    mom_5d = np.full(len(close_1d), np.nan)
    for i in range(5, len(close_1d)):
        mom_5d[i] = close_1d[i] - close_1d[i-5]
    
    # Align WoW momentum to 6h timeframe
    mom_5d_aligned = align_htf_to_ltf(prices, df_1d, mom_5d)
    
    # Calculate 6h RSI(14) for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    for i in range(n):
        if i < 14:
            if i > 0:
                avg_gain[i] = np.mean(gain[:i+1])
                avg_loss[i] = np.mean(loss[:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (20-period)
    vol_ma20 = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            if i > 0:
                vol_ma20[i] = np.mean(volume[:i+1])
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(mom_5d_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Positive WoW momentum + RSI > 50 + volume spike
            if (mom_5d_aligned[i] > 0 and 
                rsi[i] > 50 and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Negative WoW momentum + RSI < 50 + volume spike
            elif (mom_5d_aligned[i] < 0 and 
                  rsi[i] < 50 and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Momentum turns negative or RSI < 40
            if (mom_5d_aligned[i] <= 0 or 
                rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Momentum turns positive or RSI > 60
            if (mom_5d_aligned[i] >= 0 or 
                rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals