#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Pivot_R1_S1_Breakout_Volume_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation (same timeframe)
    df_1d = prices.copy()  # Already daily data
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate daily pivot points (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    
    # Calculate weekly EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1d[i]) or np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or \
           np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema34_1w = ema34_1w_aligned[i]
        pivot = pivot_1d[i]
        r1 = r1_1d[i]
        s1 = s1_1d[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        # Trend filter: price above/below weekly EMA34
        uptrend = price > ema34_1w
        downtrend = price < ema34_1w
        
        if position == 0:
            # Long: Price breaks above R1 + volume + uptrend
            if price > r1 and volume_confirmed and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + volume + downtrend
            elif price < s1 and volume_confirmed and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below pivot
            if price < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above pivot
            if price > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals