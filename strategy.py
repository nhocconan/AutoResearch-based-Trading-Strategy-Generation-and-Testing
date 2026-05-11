# 1d_Weekly_Pivot_Breakout_Trend_Filter
# Hypothesis: Weekly pivot levels act as strong support/resistance. Breaking above/below R2/S2 with trend alignment and volume capture directional moves in both bull and bear markets. Uses weekly pivot with daily trend filter and volume confirmation to reduce false signals.
# Target: 30-100 trades over 4 years on 1d timeframe. Focus on BTC/ETH/USD pairs.

#!/usr/bin/env python3
name = "1d_Weekly_Pivot_Breakout_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtr_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly and daily data
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 10 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly pivot points (using previous week)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    
    # Daily trend filter (EMA50 > EMA200)
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_up_1d = ema50_1d > ema200_1d
    trend_down_1d = ema50_1d < ema200_1d
    
    # Align all to daily
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R2 in daily uptrend with volume surge
            if (close[i] > r2_aligned[i] and 
                trend_up_aligned[i] and 
                volume[i] > 1.8 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 in daily downtrend with volume surge
            elif (close[i] < s2_aligned[i] and 
                  trend_down_aligned[i] and 
                  volume[i] > 1.8 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below R1 or daily trend changes
            if (close[i] < r1_aligned[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above S1 or daily trend changes
            if (close[i] > s1_aligned[i] or not trend_down_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals