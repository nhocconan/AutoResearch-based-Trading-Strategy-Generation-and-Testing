#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 12h context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR(14) for volatility and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(n):
        if i < 14:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    # Calculate daily range for adaptive thresholds
    daily_range = df_1d['high'].values - df_1d['low'].values
    daily_range_avg = np.full(len(df_1d), np.nan)
    for i in range(5, len(df_1d)):
        daily_range_avg[i] = np.mean(daily_range[i-5:i+1])
    daily_range_avg_aligned = align_htf_to_ltf(prices, df_1d, daily_range_avg)
    
    signals = np.zeros(n)
    position = 0
    
    # Start after warmup period
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(daily_range_avg_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume filter: require above average volume
        volume_filter = vol_ratio > 1.5
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr[i] > daily_range_avg_aligned[i] * 0.02
        
        if position == 0:
            # Enter long on strong upward momentum with volume
            if (price > close[i-1] and 
                close[i-1] > close[i-2] and
                volume_filter and vol_filter):
                signals[i] = 0.25
                position = 1
            # Enter short on strong downward momentum with volume
            elif (price < close[i-1] and 
                  close[i-1] < close[i-2] and
                  volume_filter and vol_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on momentum reversal or volatility drop
            if (price < close[i-1] or 
                atr[i] < daily_range_avg_aligned[i] * 0.01):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on momentum reversal or volatility drop
            if (price > close[i-1] or 
                atr[i] < daily_range_avg_aligned[i] * 0.01):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Momentum_Volume_Filter"
timeframe = "12h"
leverage = 1.0