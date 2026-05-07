#!/usr/bin/env python3
# 1D_WeeklyTrend_VolumeBreakout_v2
# Hypothesis: Daily timeframe strategy using weekly trend (EMA10) and daily breakouts from weekly high/low with volume confirmation.
# Works in bull markets (breakouts above weekly high in uptrend) and bear markets (breakdowns below weekly low in downtrend).
# Volume filter reduces false breakouts. Target 10-25 trades/year to minimize fee drag.

name = "1D_WeeklyTrend_VolumeBreakout_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and reference levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need enough data for EMA10
        return np.zeros(n)
    
    # Calculate weekly EMA10 for trend filter
    weekly_close = df_1w['close'].values
    ema_10 = pd.Series(weekly_close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Weekly high and low for breakout levels
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align weekly data to daily timeframe
    ema_10_aligned = align_htf_to_ltf(prices, df_1w, ema_10)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume filter: current volume > 1.5x average volume (20-day)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Ensure we have volume MA and EMA10 data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_10_aligned[i]) or np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (1.5x average volume)
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above weekly high + uptrend (price > EMA10) + volume
            if (close[i] > weekly_high_aligned[i] and 
                close[i] > ema_10_aligned[i] and   # Uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly low + downtrend (price < EMA10) + volume
            elif (close[i] < weekly_low_aligned[i] and 
                  close[i] < ema_10_aligned[i] and   # Downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit when price returns to weekly EMA10 (trend exhaustion)
            if (position == 1 and close[i] < ema_10_aligned[i]) or \
               (position == -1 and close[i] > ema_10_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals