#!/usr/bin/env python3
name = "6h_12h_Pivot_Rotation_Squeeze_Breakout"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    sma50_12h = pd.Series(close_12h).rolling(window=50, min_periods=50).mean().values
    trend_up_12h = close_12h > sma50_12h
    
    # Get 1d data for pivot calculation (daily high/low/close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points and levels (PP, R1, S1, R2, S2)
    pivot = np.zeros(len(high_1d))
    r1 = np.zeros(len(high_1d))
    s1 = np.zeros(len(high_1d))
    r2 = np.zeros(len(high_1d))
    s2 = np.zeros(len(high_1d))
    
    for i in range(len(high_1d)):
        if i < 1:
            pivot[i] = np.nan
            r1[i] = np.nan
            s1[i] = np.nan
            r2[i] = np.nan
            s2[i] = np.nan
        else:
            # Previous day's values
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            pivot[i] = (prev_high + prev_low + prev_close) / 3.0
            r1[i] = 2 * pivot[i] - prev_low
            s1[i] = 2 * pivot[i] - prev_high
            r2[i] = pivot[i] + (prev_high - prev_low)
            s2[i] = pivot[i] - (prev_high - prev_low)
    
    # Align pivot levels to 6h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_12h_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_12h_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h_aligned = align_htf_to_ltf(prices, df_1d, s2)
    trend_up_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_up_12h)
    
    # Volatility squeeze indicator (6-period ATR / 50-period SMA ATR)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr6 = pd.Series(tr).rolling(window=6, min_periods=6).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    squeeze = atr6 / (atr50 + 1e-10)  # Avoid division by zero
    squeeze_threshold = 0.6  # Squeeze when short-term ATR < 60% of long-term ATR
    
    # Volume moving average (20-period) for confirmation
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_12h_aligned[i]) or 
            np.isnan(r1_12h_aligned[i]) or
            np.isnan(s1_12h_aligned[i]) or
            np.isnan(r2_12h_aligned[i]) or
            np.isnan(s2_12h_aligned[i]) or
            np.isnan(trend_up_12h_aligned[i]) or
            np.isnan(squeeze[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above R1 during uptrend + volatility expansion + volume
            if (close[i] > r1_12h_aligned[i] and 
                trend_up_12h_aligned[i] and 
                squeeze[i] > squeeze_threshold and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 during downtrend + volatility expansion + volume
            elif (close[i] < s1_12h_aligned[i] and 
                  not trend_up_12h_aligned[i] and 
                  squeeze[i] > squeeze_threshold and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: breakdown below pivot or trend reversal or volatility contraction
            if (close[i] < pivot_12h_aligned[i] or 
                not trend_up_12h_aligned[i] or 
                squeeze[i] < squeeze_threshold * 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: breakout above pivot or trend reversal or volatility contraction
            if (close[i] > pivot_12h_aligned[i] or 
                trend_up_12h_aligned[i] or 
                squeeze[i] < squeeze_threshold * 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals