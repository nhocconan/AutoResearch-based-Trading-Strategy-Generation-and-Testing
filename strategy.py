#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Pivot R1/S1 breakout with volume confirmation and 1d EMA34 trend filter.
# Long when price breaks above 12h R1 (pivot resistance) with volume > 1.5x 12-period average and price > 1d EMA34.
# Short when price breaks below 12h S1 (pivot support) with volume > 1.5x 12-period average and price < 1d EMA34.
# Exit when price crosses back through pivot point.
# Uses 12h pivot levels for structure, volume surge for conviction, 1d EMA34 for trend filter.
# Designed for ~15-25 trades/year per target.
name = "12h_1dPivot_R1S1_Breakout_Volume_EMA34Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for pivot levels
    df_12h = get_htf_data(prices, '12h')
    
    # Pivot points on 12h (standard floor trader's pivot)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    r1_12h = 2 * pivot_12h - low_12h
    s1_12h = 2 * pivot_12h - high_12h
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5 * 12-period average (12 * 12h = 6 days)
    vol_ma_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_filter = volume > (1.5 * vol_ma_12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_12h_aligned[i]) or np.isnan(r1_12h_aligned[i]) or
            np.isnan(s1_12h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_12[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pivot_val = pivot_12h_aligned[i]
        r1_val = r1_12h_aligned[i]
        s1_val = s1_12h_aligned[i]
        ema_val = ema_34_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume surge and above 1d EMA34
            if close_val > r1_val and vol_filter and close_val > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume surge and below 1d EMA34
            elif close_val < s1_val and vol_filter and close_val < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below pivot
            if close_val < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above pivot
            if close_val > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals