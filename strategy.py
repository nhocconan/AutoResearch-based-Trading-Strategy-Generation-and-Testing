#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Camarilla pivot (R1/S1) breakout + volume confirmation + 4h EMA34 trend filter.
Long when price breaks above R1 with volume > 1.5x 20-period volume average and close > EMA34.
Short when price breaks below S1 with volume > 1.5x 20-period volume average and close < EMA34.
Exits on opposite Camarilla level touch (S1 for longs, R1 for shorts) or EMA34 cross.
Designed to capture intraday momentum within the daily pivot structure, working in both bull and bear markets by filtering with EMA34 trend.
"""

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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (R1, S1, R2, S2, R3, S3, R4, S4)
    def camarilla_levels(high_vals, low_vals, close_vals):
        # Typical price for pivot
        pp = (high_vals + low_vals + close_vals) / 3.0
        range_val = high_vals - low_vals
        # Camarilla levels
        r1 = pp + range_val * 1.0 / 12.0
        s1 = pp - range_val * 1.0 / 12.0
        r2 = pp + range_val * 2.0 / 12.0
        s2 = pp - range_val * 2.0 / 12.0
        r3 = pp + range_val * 3.0 / 12.0
        s3 = pp - range_val * 3.0 / 12.0
        r4 = pp + range_val * 4.0 / 12.0
        s4 = pp - range_val * 4.0 / 12.0
        return r1, s1, r2, s2, r3, s3, r4, s4
    
    r1_1d, s1_1d, r2_1d, s2_1d, r3_1d, s3_1d, r4_1d, s4_1d = camarilla_levels(high_1d, low_1d, close_1d)
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(34)
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume average on 4h
    vol_ma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (4h)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # need enough for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and above EMA34
            if (close[i] > r1_1d_aligned[i] and 
                volume_confirmed and 
                close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below EMA34
            elif (close[i] < s1_1d_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema_34_4h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches S1 or closes below EMA34
            if close[i] <= s1_1d_aligned[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches R1 or closes above EMA34
            if close[i] >= r1_1d_aligned[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dCamarilla_R1S1_Breakout_Volume_EMA34Filter"
timeframe = "4h"
leverage = 1.0