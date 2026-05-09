#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA for trend (40-period)
    ema40_1w = pd.Series(df_1w['close']).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Calculate weekly pivot points
    # Typical price = (H + L + C) / 3
    tp = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # Pivot point
    pp = tp
    # Support and resistance levels
    r1 = 2 * pp - df_1w['low']
    s1 = 2 * pp - df_1w['high']
    r2 = pp + (df_1w['high'] - df_1w['low'])
    s2 = pp - (df_1w['high'] - df_1w['low'])
    r3 = df_1w['high'] + 2 * (pp - df_1w['low'])
    s3 = df_1w['low'] - 2 * (df_1w['high'] - pp)
    
    # Align pivot levels to daily timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2.values)
    
    # Volume filter: current daily volume > 1.5 * 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(40, 20)  # EMA40 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema40_1w_aligned[i]) or
            np.isnan(pp_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema40_val = ema40_1w_aligned[i]
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: price above R1 and above weekly EMA trend + volume filter
            if close[i] > r1_val and close[i] > ema40_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price below S1 and below weekly EMA trend + volume filter
            elif close[i] < s1_val and close[i] < ema40_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below pivot point
            if close[i] < pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above pivot point
            if close[i] > pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals