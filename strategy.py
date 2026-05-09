#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Trend_With_Volume_Spike_Filter"
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
    
    # Get 1d data for trend filter and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA on daily close for trend direction
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # sum of |diff| over 10 periods
    # Handle the array operations properly
    change_full = np.concatenate([np.full(10, np.nan), change])
    volatility_full = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility_full != 0, change_full / volatility_full, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama_1d = np.full_like(close_1d, np.nan)
    kama_1d[10] = close_1d[10]  # Start after 10 periods
    for i in range(11, len(close_1d)):
        if np.isnan(kama_1d[i-1]):
            kama_1d[i] = close_1d[i]
        else:
            kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
    
    # Align KAMA to 4h
    kama_4h = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Daily ATR for volatility filter (use 14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14[i] = np.mean(tr[1:15])  # Simple average of first 14
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_4h[i]) or np.isnan(atr_14_4h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Price above KAMA with volume spike
            if close[i] > kama_4h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA with volume spike
            elif close[i] < kama_4h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below KAMA OR volatility filter fails
            if close[i] < kama_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above KAMA OR volatility filter fails
            if close[i] > kama_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals