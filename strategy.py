#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 10-period ATR (volatility regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily ATR(10) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:],
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_10_1d = np.full(len(close_1d), np.nan)
    for i in range(10, len(close_1d)):
        if i == 10:
            atr_10_1d[i] = np.mean(tr_1d[1:11])
        else:
            atr_10_1d[i] = (atr_10_1d[i-1] * 9 + tr_1d[i]) / 10
    
    atr_10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    
    # Calculate 4-period volume average for volume filter
    vol_ma = np.full(n, np.nan)
    vol_period = 4
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Calculate 4-period high/low for breakout levels
    high_4 = np.full(n, np.nan)
    low_4 = np.full(n, np.nan)
    for i in range(4, n):
        high_4[i] = np.max(high[i-4:i])
        low_4[i] = np.min(low[i-4:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(10, vol_period, 4) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(atr_10_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(high_4[i]) or np.isnan(low_4[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volatility filter: avoid extreme volatility (ATR > 1.5x median)
        vol_filter = atr_10_1d_aligned[i] < np.nanmedian(atr_10_1d_aligned[:i+1]) * 1.5
        
        if position == 0:
            # Long: Break above 4-period high with volume and normal volatility
            if price > high_4[i] and vol_ratio > 1.8 and vol_filter:
                signals[i] = size
                position = 1
            # Short: Break below 4-period low with volume and normal volatility
            elif price < low_4[i] and vol_ratio > 1.8 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below 4-period low or volatility spike
            if price < low_4[i] or atr_10_1d_aligned[i] > np.nanmedian(atr_10_1d_aligned[:i+1]) * 2.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above 4-period high or volatility spike
            if price > high_4[i] or atr_10_1d_aligned[i] > np.nanmedian(atr_10_1d_aligned[:i+1]) * 2.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Breakout_4Period_VolumeVolatilityFilter"
timeframe = "4h"
leverage = 1.0