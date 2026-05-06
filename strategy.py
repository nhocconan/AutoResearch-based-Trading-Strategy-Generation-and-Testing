#!/usr/bin/env python3
# 12h_1dCamarilla_R1S1_Breakout_1dEMA34_Trend_Volume
# Uses daily Camarilla pivot levels (R1/S1) as breakout levels with daily trend filter (EMA34)
# and daily volume confirmation. Designed for 12h timeframe to capture major pivot breaks
# with trend alignment. Works in both bull and bear markets by following the daily trend.
# Features: 12h timeframe for lower trade frequency, volume spike filter, and minimum holding period.
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing.

name = "12h_1dCamarilla_R1S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "12h"
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
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (R1/S1 - tighter range for fewer trades)
    r1 = pp + range_1d * 1.1 / 12
    s1 = pp - range_1d * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily volume filter (20-period MA) with threshold for spike
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma_20)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: break above R1 with uptrend (price > EMA34) and volume spike
            if close[i] > r1_12h[i] and close[i] > ema_34_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: break below S1 with downtrend (price < EMA34) and volume spike
            elif close[i] < s1_12h[i] and close[i] < ema_34_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit conditions: price returns to EMA34 or breaks below S1
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry >= 2 and (close[i] < ema_34_12h[i] or close[i] < s1_12h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions: price returns to EMA34 or breaks above R1
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry >= 2 and (close[i] > ema_34_12h[i] or close[i] > r1_12h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals