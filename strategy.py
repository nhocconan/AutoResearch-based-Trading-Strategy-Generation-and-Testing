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
    
    # Load 1d data for weekly pivot levels (Friday close) and EMA trend (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from previous Friday (5 trading days ago)
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    idx = np.arange(len(close_1d))
    prev_friday_idx = np.maximum(idx - 5, 0)  # 5 trading days back
    prev_high = high_1d[prev_friday_idx]
    prev_low = low_1d[prev_friday_idx]
    prev_close = close_1d[prev_friday_idx]
    pivot_range = prev_high - prev_low
    r1 = prev_close + 1.1 * pivot_range / 12
    s1 = prev_close - 1.1 * pivot_range / 12
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 15-period average
    vol_avg_15 = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_15[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above weekly R1 + above 1d EMA50 + volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > 1.5 * vol_avg_15[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below weekly S1 + below 1d EMA50 + volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > 1.5 * vol_avg_15[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses 1d EMA50 in opposite direction
            if position == 1:
                # Exit long: Close below 1d EMA50
                if close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Close above 1d EMA50
                if close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_WeeklyPivot_R1_S1_Breakout_1D_EMA50_Trend_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0