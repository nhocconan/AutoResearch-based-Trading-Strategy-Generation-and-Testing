#!/usr/bin/env python3
# 4h_1dPivot_HighLow_Breakout_Trend_Volume
# Uses daily pivot high/low (breakout of previous day's high/low) with daily EMA50 trend filter and volume confirmation.
# Designed for 4h timeframe to capture momentum breaks aligned with daily trend.
# Works in bull/bear markets by following daily trend direction.
# Target: 75-200 total trades over 4 years with 0.30 position sizing.

name = "4h_1dPivot_HighLow_Breakout_Trend_Volume"
timeframe = "4h"
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
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Previous day's high and low (breakout levels)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_high[0] = np.nan  # First day has no previous
    prev_low[0] = np.nan
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume average (20-period) for volume confirmation
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to 4h timeframe
    prev_high_4h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_4h = align_htf_to_ltf(prices, df_1d, prev_low)
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_4h = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(prev_high_4h[i]) or np.isnan(prev_low_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ma_20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Volume condition: current volume > 1.5x daily 20-period MA
        vol_condition = volume[i] > (1.5 * vol_ma_20_4h[i])
        
        if position == 0:
            # Long: break above previous day's high with uptrend (price > EMA50) and volume
            if close[i] > prev_high_4h[i] and close[i] > ema_50_4h[i] and vol_condition:
                signals[i] = 0.30
                position = 1
                bars_since_entry = 0
            # Short: break below previous day's low with downtrend (price < EMA50) and volume
            elif close[i] < prev_low_4h[i] and close[i] < ema_50_4h[i] and vol_condition:
                signals[i] = -0.30
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price returns below EMA50 or breaks below previous day's low
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry >= 2 and (close[i] < ema_50_4h[i] or close[i] < prev_low_4h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price returns above EMA50 or breaks above previous day's high
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry >= 2 and (close[i] > ema_50_4h[i] or close[i] > prev_high_4h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.30
    
    return signals