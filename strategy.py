#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_v1
# Hypothesis: Price breaks above weekly R3 pivot or below weekly S3 pivot with volume confirmation on 6h timeframe.
# Uses weekly pivot levels as strong support/resistance, with volume surge to confirm breakout strength.
# Works in both bull and bear markets by trading breakouts in the direction of the weekly pivot bias.
# Target: 15-30 trades/year with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R3 = H + 2*(Pivot - L)
    # S3 = L - 2*(H - Pivot)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r3 = weekly_high + 2.0 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2.0 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = vol_ma_period + 5  # Need volume MA and weekly data
    
    for i in range(start_idx, n):
        # Skip if weekly data not yet available (first few weeks)
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below weekly pivot (mean reversion) OR volume drops significantly
            if close[i] < pivot_aligned[i] or volume[i] < 0.5 * vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above weekly pivot OR volume drops significantly
            if close[i] > pivot_aligned[i] or volume[i] < 0.5 * vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above weekly R3 with volume surge
            if close[i] > r3_aligned[i] and vol_surge[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below weekly S3 with volume surge
            elif close[i] < s3_aligned[i] and vol_surge[i]:
                position = -1
                signals[i] = -0.25
    
    return signals