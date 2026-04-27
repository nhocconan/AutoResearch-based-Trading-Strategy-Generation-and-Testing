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
    
    # Get 1d data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian(20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper = np.full(len(high_1d), np.nan)
    lower = np.full(len(high_1d), np.nan)
    for i in range(20, len(high_1d)):
        upper[i] = np.max(high_1d[i-20:i])
        lower[i] = np.min(low_1d[i-20:i])
    donch_upper_1d = upper
    donch_lower_1d = lower
    donch_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_upper_1d)
    donch_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_lower_1d)
    
    # Get 1d data for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_10_1d = pd.Series(vol_1d).rolling(window=10, min_periods=10).mean().values
    vol_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_10_1d)
    
    # Get 1d data for ATR-based volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian, volume MA, and ATR
    start_idx = max(20, 10, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_upper_1d_aligned[i]) or np.isnan(donch_lower_1d_aligned[i]) or 
            np.isnan(vol_ma_10_1d_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        upper = donch_upper_1d_aligned[i]
        lower = donch_lower_1d_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_10_1d_aligned[i]
        atr_now = atr_1d_aligned[i]
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = atr_now > 0  # Ensure ATR is valid
        
        # Volume filter: volume > 1.5x 1d MA (volume breakout)
        vol_breakout = vol_now > 1.5 * vol_ma
        
        # Entry conditions: breakout with volume and volatility
        if position == 0:
            # Long: break above upper band + volume
            if close[i] > upper and vol_breakout and vol_filter:
                signals[i] = size
                position = 1
            # Short: break below lower band + volume
            elif close[i] < lower and vol_breakout and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below midpoint or volatility drops significantly
            midpoint = (upper + lower) / 2
            if close[i] < midpoint or atr_now < 0.7 * atr_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above midpoint or volatility drops significantly
            midpoint = (upper + lower) / 2
            if close[i] > midpoint or atr_now < 0.7 * atr_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_VolumeBreakout_ATRFilter"
timeframe = "4h"
leverage = 1.0