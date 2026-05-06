#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Pivot Point levels with daily trend filter and volume confirmation
# - Uses weekly Pivot Points (R1-R4, S1-S4) for institutional level structure
# - Uses 1d EMA34 for trend direction filter (long above, short below)
# - Uses 6h volume spike for entry confirmation
# - Enters long when price breaks above weekly R3 with 1d uptrend and volume
# - Enters short when price breaks below weekly S3 with 1d downtrend and volume
# - Exits when price returns to weekly pivot point or opposite S1/R1 level
# - Designed to capture major trend moves with institutional level respect in both bull/bear markets
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_WeeklyPivot_R3S3_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def calculate_pivot_points(high, low, close):
    """Calculate weekly Pivot Point levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Pivot Points
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Pivot Points
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    pivot_w, r1_w, r2_w, r3_w, r4_w, s1_w, s2_w, s3_w, s4_w = calculate_pivot_points(high_w, low_w, close_w)
    
    # Align weekly Pivot Points to 6h timeframe
    pivot_w_6h = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_6h = align_htf_to_ltf(prices, df_w, r1_w)
    r2_w_6h = align_htf_to_ltf(prices, df_w, r2_w)
    r3_w_6h = align_htf_to_ltf(prices, df_w, r3_w)
    r4_w_6h = align_htf_to_ltf(prices, df_w, r4_w)
    s1_w_6h = align_htf_to_ltf(prices, df_w, s1_w)
    s2_w_6h = align_htf_to_ltf(prices, df_w, s2_w)
    s3_w_6h = align_htf_to_ltf(prices, df_w, s3_w)
    s4_w_6h = align_htf_to_ltf(prices, df_w, s4_w)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_w_6h[i]) or np.isnan(s3_w_6h[i]) or 
            np.isnan(pivot_w_6h[i]) or np.isnan(r1_w_6h[i]) or 
            np.isnan(s1_w_6h[i]) or np.isnan(ema_34_1d_6h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above weekly R3 with 1d uptrend and volume
            if close[i] > r3_w_6h[i] and close[i] > ema_34_1d_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S3 with 1d downtrend and volume
            elif close[i] < s3_w_6h[i] and close[i] < ema_34_1d_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly pivot OR breaks below S1
            if close[i] < pivot_w_6h[i] or close[i] < s1_w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly pivot OR breaks above R1
            if close[i] > pivot_w_6h[i] or close[i] > r1_w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals