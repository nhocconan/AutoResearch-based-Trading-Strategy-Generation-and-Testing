#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_regime_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 2)
    r4 = pivot + (range_1d * 1.1)
    s3 = pivot - (range_1d * 1.1 / 2)
    s4 = pivot - (range_1d * 1.1)
    
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Chopiness index from 1d for regime filter
    if len(df_1d) < 14:
        return np.zeros(n)
    atr_1d = []
    for i in range(len(df_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        atr_1d.append(tr)
    atr_1d = np.array(atr_1d)
    atr_ma = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    sum_tr = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr / (atr_ma * 14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    # Volume filter: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Chop regime: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending
        in_range = chop_aligned[i] > 61.8
        
        # Volume confirmation
        vol_ok = volume_filter[i]
        
        # Mean reversion in ranging market: sell at resistance, buy at support
        long_signal = in_range and vol_ok and close[i] <= s3_aligned[i]
        short_signal = in_range and vol_ok and close[i] >= r3_aligned[i]
        
        # Exit when price reaches opposite level or chop leaves range
        exit_long = (not in_range) or (close[i] >= pivot_aligned[i]) or (chop_aligned[i] < 38.2)
        exit_short = (not in_range) or (close[i] <= pivot_aligned[i]) or (chop_aligned[i] < 38.2)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals