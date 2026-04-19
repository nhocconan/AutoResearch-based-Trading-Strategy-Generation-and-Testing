#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Pivot_R1S1_Breakout_Volume_ATRFilter_Tight"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute hour filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Get 1d data for Camarilla pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    
    # Align Camarilla levels to 1h timeframe
    pivot_1h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 4h ATR for volatility filter (use 4h for less noise)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h[0] = tr1[0]
    atr_4h = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Volume confirmation: current volume > 3x 20-period average (tightened)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1h[i]) or np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or \
           np.isnan(atr_4h_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            if position != 0:
                signals[i] = 0.0  # close position outside session
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_4h_aligned[i]
        pivot = pivot_1h[i]
        r1 = r1_1h[i]
        s1 = s1_1h[i]
        
        volume_confirmed = vol > 3.0 * vol_ma
        
        if position == 0:
            # Long: Price breaks above R1 + volume
            if price > r1 and volume_confirmed:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: Price breaks below S1 + volume
            elif price < s1 and volume_confirmed:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit: Price returns below pivot OR ATR stop (1.5x ATR from entry)
            if price < pivot or price < (entry_price - 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: Price returns above pivot OR ATR stop (1.5x ATR from entry)
            if price > pivot or price > (entry_price + 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals