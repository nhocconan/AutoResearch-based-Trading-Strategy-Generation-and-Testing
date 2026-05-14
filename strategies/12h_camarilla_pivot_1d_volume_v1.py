#!/usr/bin/env python3
# 12h_camarilla_pivot_1d_volume_v1
# Hypothesis: Camarilla pivot levels from 1-day chart combined with volume confirmation and 1-week trend filter.
# Long when price touches S3 level with volume > 1.5x average and 1-week close > 1-week open.
# Short when price touches R3 level with volume > 1.5x average and 1-week close < 1-week open.
# Exit when price reaches S1/R1 levels or opposite signal.
# Designed to work in both bull and bear markets by exploiting mean reversion at extreme intraday levels.
# Target: 15-25 trades/year to minimize fee drag while capturing high-probability reversals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla pivots (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1-day bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    S1 = close_1d - (high_1d - low_1d) * 1.0833
    S2 = close_1d - (high_1d - low_1d) * 1.1666
    S3 = close_1d - (high_1d - low_1d) * 1.2500
    S4 = close_1d - (high_1d - low_1d) * 1.3333
    R1 = close_1d + (high_1d - low_1d) * 1.0833
    R2 = close_1d + (high_1d - low_1d) * 1.1666
    R3 = close_1d + (high_1d - low_1d) * 1.2500
    R4 = close_1d + (high_1d - low_1d) * 1.3333
    
    # Align 1-day Camarilla levels to 12-hour chart
    S3_1d = align_htf_to_ltf(prices, df_1d, S3)
    S1_1d = align_htf_to_ltf(prices, df_1d, S1)
    R1_1d = align_htf_to_ltf(prices, df_1d, R1)
    R3_1d = align_htf_to_ltf(prices, df_1d, R3)
    
    # Get 1-week data for trend filter (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # 1-week trend: close > open = uptrend, close < open = downtrend
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    weekly_uptrend = close_1w > open_1w
    weekly_downtrend = close_1w < open_1w
    
    # Align 1-week trend to 12-hour chart
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Volume confirmation: 24-period average (2 days of 12h data)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(S3_1d[i]) or np.isnan(S1_1d[i]) or np.isnan(R1_1d[i]) or np.isnan(R3_1d[i]) or \
           np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S1 level or opposite signal
            if close[i] <= S1_1d[i] or \
               (close[i] >= R3_1d[i] and volume[i] > 1.5 * avg_volume[i] and weekly_downtrend_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R1 level or opposite signal
            if close[i] >= R1_1d[i] or \
               (close[i] <= S3_1d[i] and volume[i] > 1.5 * avg_volume[i] and weekly_uptrend_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price touches S3 level with volume and weekly uptrend
            if close[i] <= S3_1d[i] and volume_ok and weekly_uptrend_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price touches R3 level with volume and weekly downtrend
            elif close[i] >= R3_1d[i] and volume_ok and weekly_downtrend_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals