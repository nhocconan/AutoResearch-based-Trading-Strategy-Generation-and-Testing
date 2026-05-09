#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mpf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily pivot breakout with weekly trend filter and volume confirmation
# Works in bull markets via breakout momentum, in bear via mean reversion at extreme levels
# Target: 10-25 trades/year per symbol to avoid fee drag while capturing meaningful moves

name = "1d_Pivot_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter (uses only weekly closes)
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's OHLC for standard pivot calculation
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Standard pivot point and support/resistance levels
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    r1 = 2 * pivot - prev_low_1d
    s1 = 2 * pivot - prev_high_1d
    r2 = pivot + (prev_high_1d - prev_low_1d)
    s2 = pivot - (prev_high_1d - prev_low_1d)
    
    # Align pivot levels to daily
    pivot_1d = align_htf_to_ltf(prices, df_1d, pivot)
    r1_1d = align_htf_to_ltf(prices, df_1d, r1)
    s1_1d = align_htf_to_ltf(prices, df_1d, s1)
    r2_1d = align_htf_to_ltf(prices, df_1d, r2)
    s2_1d = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume filter: above 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1d[i]) or np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or 
            np.isnan(r2_1d[i]) or np.isnan(s2_1d[i]) or np.isnan(ema_20_1d[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        if position == 0:
            # Long entry: price breaks above R1 with weekly uptrend
            if (close[i] > r1_1d[i] and 
                close[i] > ema_20_1d[i] and  # weekly uptrend
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with weekly downtrend
            elif (close[i] < s1_1d[i] and 
                  close[i] < ema_20_1d[i] and  # weekly downtrend
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below pivot
            if close[i] < pivot_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above pivot
            if close[i] > pivot_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals