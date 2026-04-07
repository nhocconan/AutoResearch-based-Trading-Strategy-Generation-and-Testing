#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_trend_volume_v1
Hypothesis: Uses Camarilla pivot levels from 1-day timeframe for entry/exit, filtered by 1-week EMA trend direction and volume confirmation. 
In uptrend (price > weekly EMA50), long at S1 (support 1) with target at R1 (resistance 1); in downtrend (price < weekly EMA50), short at R1 with target at S1.
Designed for 12h timeframe to capture swing moves with low trade frequency (<30/year) to minimize fee drag. Works in both bull and bear markets by following higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
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
    
    # Calculate 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous 1-day OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1 for completed day)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    # S1 = C - (H-L)*1.08/2
    s1 = prev_close - (range_ * 1.08 / 2)
    # R1 = C + (H-L)*1.08/2
    r1 = prev_close + (range_ * 1.08 / 2)
    
    # Align to 12h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Volume confirmation: > 1.3x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start after first bar for prev day data
        # Skip if data not available
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > (vol_ma[i] * 1.3)
        
        if position == 1:  # Long position
            # Exit: price reaches R1 or trend changes
            if high[i] >= r1_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S1 or trend changes
            if low[i] <= s1_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price at S1 in uptrend (price above weekly EMA50)
                if (low[i] <= s1_aligned[i] and 
                    close[i] > ema_50_1w_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price at R1 in downtrend (price below weekly EMA50)
                elif (high[i] >= r1_aligned[i] and 
                      close[i] < ema_50_1w_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals