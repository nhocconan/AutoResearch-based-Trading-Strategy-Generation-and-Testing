#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for pivot levels and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Previous day's OHLC for Camarilla pivot levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Key levels: S1, S2, R1, R2 (most significant)
    s1 = pivot - (range_hl * 1.1 / 12)
    s2 = pivot - (range_hl * 1.1 / 6)
    r1 = pivot + (range_hl * 1.1 / 12)
    r2 = pivot + (range_hl * 1.1 / 6)
    
    # Align all data to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period average (avoid low volatility chop)
        atr_ma_50 = pd.Series(atr_1d_aligned[:i+1]).rolling(window=50, min_periods=50).mean().iloc[-1] if i >= 99 else np.nan
        if np.isnan(atr_ma_50) or atr_1d_aligned[i] < atr_ma_50:
            signals[i] = 0.0 if position == 0 else (position_size if position == 1 else -position_size)
            continue
        
        # Entry conditions: price near S1/S2 for long, R1/R2 for short
        near_support = (close[i] <= s1_aligned[i] * 1.005) or (close[i] <= s2_aligned[i] * 1.005)
        near_resistance = (close[i] >= r1_aligned[i] * 0.995) or (close[i] >= r2_aligned[i] * 0.995)
        
        if position == 0:
            if near_support:
                position = 1
                signals[i] = position_size
            elif near_resistance:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price reaches pivot level or shows rejection
            if close[i] >= ((s1_aligned[i] + s2_aligned[i]) / 2) * 1.002:  # Midpoint of S1-S2
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price reaches pivot level or shows rejection
            if close[i] <= ((r1_aligned[i] + r2_aligned[i]) / 2) * 0.998:  # Midpoint of R1-R2
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Pivot_Volatility_Filter_v1"
timeframe = "4h"
leverage = 1.0