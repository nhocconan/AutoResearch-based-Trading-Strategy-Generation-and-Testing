#!/usr/bin/env python3
"""
12h 1-Week Price Channel Breakout with Volume and Trend Confirmation
Hypothesis: In both bull and bear markets, price tends to respect weekly high/low channels.
Breakouts above the weekly high or below the weekly low with volume confirmation and
weekly trend alignment capture significant moves while minimizing whipsaws. The 12h timeframe
provides a balance between signal quality and trade frequency, targeting 12-37 trades per year.
"""
name = "12h_WeeklyChannel_Breakout_VolumeTrend"
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
    
    # === Weekly High/Low Channel (from 1w data) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Align weekly channel to 12h timeframe (will use previous week's values)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # === Weekly Trend (EMA 34 on weekly close) ===
    weekly_ema34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema34)
    
    # === Volume Spike (20-period on 12h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Require strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure weekly EMA ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_ema34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly high + volume spike + price above weekly EMA (uptrend)
            if (close[i] > weekly_high_aligned[i] and 
                vol_spike[i] and
                close[i] > weekly_ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly low + volume spike + price below weekly EMA (downtrend)
            elif (close[i] < weekly_low_aligned[i] and 
                  vol_spike[i] and
                  close[i] < weekly_ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below weekly low OR volume drops significantly
            if close[i] < weekly_low_aligned[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly high OR volume drops significantly
            if close[i] > weekly_high_aligned[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals