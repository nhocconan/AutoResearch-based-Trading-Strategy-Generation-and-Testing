#!/usr/bin/env python3
# 12h_1w_1d_camarilla_pivot_volume_v1
# Hypothesis: 12-hour Camarilla pivot reversal with volume confirmation and weekly trend filter.
# Long: price touches S3 support AND volume > 1.5x 20-period average AND weekly close > weekly open (bullish week).
# Short: price touches R3 resistance AND volume > 1.5x 20-period average AND weekly close < weekly open (bearish week).
# Exit: price crosses pivot point (PP) or touches opposite S1/R1 level with volume confirmation.
# Designed to capture reversals at key institutional levels with strict entry criteria to limit trades (target: 15-35 trades/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-period average volume
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # 1-day Camarilla pivot levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day using previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    camarilla_multipliers = {
        'S1': 1.0/12, 'S2': 1.0/6, 'S3': 1.0/4,
        'R1': 1.0/12, 'R2': 1.0/6, 'R3': 1.0/4
    }
    
    # Arrays to store daily Camarilla levels
    S1 = np.full(len(close_1d), np.nan)
    S2 = np.full(len(close_1d), np.nan)
    S3 = np.full(len(close_1d), np.nan)
    R1 = np.full(len(close_1d), np.nan)
    R2 = np.full(len(close_1d), np.nan)
    R3 = np.full(len(close_1d), np.nan)
    PP = np.full(len(close_1d), np.nan)
    
    # Calculate from index 1 onwards (need previous day)
    for i in range(1, len(close_1d)):
        # Previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Pivot point
        pp = (ph + pl + pc) / 3
        PP[i] = pp
        
        # Range
        range_val = ph - pl
        
        # Support levels
        S1[i] = pp - range_val * camarilla_multipliers['S1']
        S2[i] = pp - range_val * camarilla_multipliers['S2']
        S3[i] = pp - range_val * camarilla_multipliers['S3']
        
        # Resistance levels
        R1[i] = pp + range_val * camarilla_multipliers['R1']
        R2[i] = pp + range_val * camarilla_multipliers['R2']
        R3[i] = pp + range_val * camarilla_multipliers['R3']
    
    # Align Camarilla levels to 12h timeframe
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    S2_12h = align_htf_to_ltf(prices, df_1d, S2)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    R2_12h = align_htf_to_ltf(prices, df_1d, R2)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    PP_12h = align_htf_to_ltf(prices, df_1d, PP)
    
    # Weekly trend filter (bullish/bearish week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    
    # Weekly bullish/bearish (1 = bullish week, -1 = bearish week)
    weekly_trend = np.where(close_1w > open_1w, 1, -1)
    weekly_trend_12h = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start from 20 to have volume average
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Get current Camarilla levels
        s1 = S1_12h[i]
        s2 = S2_12h[i]
        s3 = S3_12h[i]
        r1 = R1_12h[i]
        r2 = R2_12h[i]
        r3 = R3_12h[i]
        pp = PP_12h[i]
        weekly_trend_val = weekly_trend_12h[i]
        
        # Skip if any required value is NaN
        if np.isnan(avg_vol) or np.isnan(s1) or np.isnan(s2) or np.isnan(s3) or \
           np.isnan(r1) or np.isnan(r2) or np.isnan(r3) or np.isnan(pp) or \
           np.isnan(weekly_trend_val):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = vol > 1.5 * avg_vol
        
        if position == 1:  # Long position
            # Exit: price crosses PP or touches R1 with volume
            if price < pp or (price > r1 and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses PP or touches S1 with volume
            if price > pp or (price < s1 and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: price touches S3 support with volume in bullish week
            if abs(price - s3) < 0.001 * s3 and vol_surge and weekly_trend_val == 1:
                position = 1
                signals[i] = 0.25
            # Short: price touches R3 resistance with volume in bearish week
            elif abs(price - r3) < 0.001 * r3 and vol_surge and weekly_trend_val == -1:
                position = -1
                signals[i] = -0.25
    
    return signals