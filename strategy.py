#!/usr/bin/env python3
# 1d_1w_Camarilla_Pivot_R3_S3_Breakout_Trend_Filter_Volume
# Uses weekly Camarilla pivot levels (R3/S3) as breakout levels with 1d trend filter and volume confirmation.
# Long when price breaks above weekly R3 in uptrend with volume spike.
# Short when price breaks below weekly S3 in downtrend with volume spike.
# Exit when price crosses back through the 1d EMA34.
# Designed for 1d timeframe to capture institutional levels with trend alignment.

name = "1d_1w_Camarilla_Pivot_R3_S3_Breakout_Trend_Filter_Volume"
timeframe = "1d"
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
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    # Formula: Pivot = (H + L + C) / 3
    # R3 = Pivot + (H - L) * 1.1
    # S3 = Pivot - (H - L) * 1.1
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r3_1w = pivot_1w + (high_1w - low_1w) * 1.1
    s3_1w = pivot_1w - (high_1w - low_1w) * 1.1
    
    # Align weekly Camarilla levels to daily timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA34 to daily timeframe (no alignment needed, but for consistency)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily volume filter (20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    for i in range(34, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above weekly R3 with uptrend and volume spike
            if close[i] > r3_1w_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below weekly S3 with downtrend and volume spike
            elif close[i] < s3_1w_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price crosses back below EMA34
            # Minimum holding period of 2 days to reduce churn
            if bars_since_entry >= 2 and close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses back above EMA34
            # Minimum holding period of 2 days to reduce churn
            if bars_since_entry >= 2 and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals