#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
# Hypothesis: Price reacts strongly to Camarilla pivot levels (R1/S1) on the daily timeframe.
# In trending markets (identified by 1-day EMA34), breaks of R1/S1 with volume confirmation
# capture strong continuation moves. Works in both bull and bear markets by following the
# 1-day trend direction. Uses volume spike to filter low-conviction breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
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
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day (use shift(1) to avoid look-ahead)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift to get previous day's values (avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day will have invalid values (rolled from last) - handle with valid mask
    valid_prev = np.arange(len(close_1d)) > 0  # Skip first day
    
    # Calculate Camarilla levels for previous day
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 12h timeframe (wait for 12h bar to close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation (2-period MA on 12h chart = 1 day)
    volume_ma = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34_1d (34), volume MA (2)
    start_idx = max(34, 2)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter (using valid previous day data)
        day_idx = i // 2  # Approximate mapping: 2x 12h bars per day
        if day_idx >= len(ema_34_1d) or day_idx < 1:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        uptrend = close[i] > ema_34_1d[day_idx]
        downtrend = close[i] < ema_34_1d[day_idx]
        
        # Volume confirmation (volume > 1.5x average)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + break above R1 + volume
            if uptrend and close[i] > R1_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + break below S1 + volume
            elif downtrend and close[i] < S1_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters R1-S1 range
            if not uptrend or close[i] < R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters R1-S1 range
            if not downtrend or close[i] > S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals