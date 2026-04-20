#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load 1d data for daily pivot
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot and support/resistance levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    # R1 = pivot + range/3, S1 = pivot - range/3 (more sensitive)
    r1_1d = pivot_1d + range_1d / 3.0
    s1_1d = pivot_1d - range_1d / 3.0
    
    # Align to 6h (previous day's levels available at 6h open)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 6h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 24-period average (4 days of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: break above R1 with volume, above weekly EMA50 (uptrend)
            if price > r1_1d_aligned[i] and price > ema_50_1w_aligned[i] and vol > 1.3 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below S1 with volume, below weekly EMA50 (downtrend)
            elif price < s1_1d_aligned[i] and price < ema_50_1w_aligned[i] and vol > 1.3 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below S1 (mean reversion) or weekly trend fails
            if price < s1_1d_aligned[i] or price < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above R1 (mean reversion) or weekly trend fails
            if price > r1_1d_aligned[i] or price > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyTrend_DailyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0