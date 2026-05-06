#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla pivot levels with volume confirmation and 12h EMA34 trend filter
# - Long when price crosses above Camarilla R1 with volume expansion and price above 12h EMA34
# - Short when price crosses below Camarilla S1 with volume expansion and price below 12h EMA34
# - Exit when price returns to Camarilla pivot (midpoint of high-low)
# - Volume filter requires current volume > 1.5x 20-period average
# - Designed to capture reversals at key levels in ranging markets while filtering with trend
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "4h_CamarillaPivot_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla and EMA calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (based on previous day's range)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot and ranges
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r1_12h = close_12h + (range_12h * 1.1 / 12)
    s1_12h = close_12h - (range_12h * 1.1 / 12)
    r2_12h = close_12h + (range_12h * 1.1 / 6)
    s2_12h = close_12h - (range_12h * 1.1 / 6)
    r3_12h = close_12h + (range_12h * 1.1 / 4)
    s3_12h = close_12h - (range_12h * 1.1 / 4)
    r4_12h = close_12h + (range_12h * 1.1 / 2)
    s4_12h = close_12h - (range_12h * 1.1 / 2)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h indicators to 4h timeframe
    pivot_12h_4h = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_4h = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_4h = align_htf_to_ltf(prices, df_12h, s1_12h)
    ema_34_12h_4h = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(pivot_12h_4h[i]) or np.isnan(r1_12h_4h[i]) or np.isnan(s1_12h_4h[i]) or 
            np.isnan(ema_34_12h_4h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price crosses above R1 with volume expansion and above EMA34
            if close[i] > r1_12h_4h[i] and close[i-1] <= r1_12h_4h[i-1] and volume_filter[i] and close[i] > ema_34_12h_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below S1 with volume expansion and below EMA34
            elif close[i] < s1_12h_4h[i] and close[i-1] >= s1_12h_4h[i-1] and volume_filter[i] and close[i] < ema_34_12h_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot level
            if close[i] <= pivot_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot level
            if close[i] >= pivot_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals