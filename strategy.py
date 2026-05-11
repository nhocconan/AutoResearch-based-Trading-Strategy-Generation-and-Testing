#!/usr/bin/env python3
name = "12h_20Week_HighLow_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_20w = get_htf_data(prices, '20w')  # 20-week high/low from daily resampled
    
    if len(df_1d) < 50 or len(df_20w) < 20:
        return np.zeros(n)
    
    # Calculate 20-week high and low from daily data (using 100 trading days)
    # 20 weeks * 5 days = 100 days
    close_1d = df_1d['close'].values
    high_20w = np.full(len(close_1d), np.nan)
    low_20w = np.full(len(close_1d), np.nan)
    
    for i in range(100, len(close_1d)):
        high_20w[i] = np.max(close_1d[i-100:i])
        low_20w[i] = np.min(close_1d[i-100:i])
    
    # Daily trend filter (EMA50 > EMA200)
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_up_1d = ema50_1d > ema200_1d
    trend_down_1d = ema50_1d < ema200_1d
    
    # Align all to 12h
    high_20w_aligned = align_htf_to_ltf(prices, df_1d, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1d, low_20w)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 20-week high in daily uptrend with volume surge
            if (close[i] > high_20w_aligned[i] and 
                trend_up_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-week low in daily downtrend with volume surge
            elif (close[i] < low_20w_aligned[i] and 
                  trend_down_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below 20-week low or trend changes
            if (close[i] < low_20w_aligned[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above 20-week high or trend changes
            if (close[i] > high_20w_aligned[i] or not trend_down_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals