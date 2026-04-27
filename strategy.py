# 2025-06-24: 12h_Donchian20_VolumeBreakout_ATRFilter_v2
# Hypothesis: 12h Donchian(20) breakout with volume confirmation (>1.3x 1d MA) and volatility filter (ATR > 0) works in both bull and bear markets.
# Why it should work: Breakouts capture momentum; volume filter ensures institutional participation; volatility filter avoids low-volatility whipsaws.
# Timeframe: 12h allows fewer trades (target 20-50/year) to reduce fee drag. Uses 1d for volume and volatility filters to align with institutional cycles.
# Risk: Uses midpoint exit to avoid missing trends; size 0.25 limits drawdown.

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
    
    # Get 12h data for Donchian channel
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    upper = np.full(len(high_12h), np.nan)
    lower = np.full(len(high_12h), np.nan)
    for i in range(20, len(high_12h)):
        upper[i] = np.max(high_12h[i-20:i])
        lower[i] = np.min(low_12h[i-20:i])
    donch_upper_12h = upper
    donch_lower_12h = lower
    donch_upper_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_upper_12h)
    donch_lower_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_lower_12h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Get 1d data for ATR-based volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian, volume MA, and ATR
    start_idx = max(20, 20, 14)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_upper_12h_aligned[i]) or np.isnan(donch_lower_12h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        upper = donch_upper_12h_aligned[i]
        lower = donch_lower_12h_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        atr_now = atr_1d_aligned[i]
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = atr_now > 0  # Ensure ATR is valid
        
        # Volume filter: volume > 1.3x 1d MA (volume breakout)
        vol_breakout = vol_now > 1.3 * vol_ma
        
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

name = "12h_Donchian20_VolumeBreakout_ATRFilter_v2"
timeframe = "12h"
leverage = 1.0