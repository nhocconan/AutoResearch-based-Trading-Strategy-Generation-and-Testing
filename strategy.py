#!/usr/bin/env python3
# 12h_Camarilla_Pivot_1d_trend_volume_v1
# Hypothesis: Trade Camarilla pivot level touches on 12h timeframe with 1d trend filter and volume confirmation.
# Long when price touches S3 in uptrend, short when price touches R3 in downtrend.
# Works in both bull and bear markets by fading extremes within the trend direction.
# Target: 15-25 trades/year (60-100 total over 4 years).

name = "12h_Camarilla_Pivot_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA trend filter (50-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 12h Camarilla pivot levels (based on previous day)
    # Calculate daily OHLC from 1d data for pivot calculation
    if len(df_1d) < 2:
        return np.zeros(n)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_S3 = np.full(len(df_1d), np.nan)
    camarilla_R3 = np.full(len(df_1d), np.nan)
    for i in range(1, len(df_1d)):
        # Previous day's range
        prev_range = daily_high[i-1] - daily_low[i-1]
        camarilla_S3[i] = daily_close[i-1] - 1.1 * prev_range / 6
        camarilla_R3[i] = daily_close[i-1] + 1.1 * prev_range / 6
    
    # Align Camarilla levels to 12h timeframe
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Start from sufficient lookback
    start_idx = max(50, 20) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S1 (mean reversion target) or trend breaks down
            camarilla_S1 = camarilla_S3_aligned[i] + 1.1 * (daily_close[-1] - daily_low[-1]) / 6 if len(daily_close) > 0 else 0
            if close[i] >= camarilla_S1 or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R1 (mean reversion target) or trend breaks up
            camarilla_R1 = camarilla_R3_aligned[i] - 1.1 * (daily_high[-1] - daily_close[-1]) / 6 if len(daily_high) > 0 else 0
            if close[i] <= camarilla_R1 or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches S3 in uptrend with volume confirmation
            if (close[i] <= camarilla_S3_aligned[i] * 1.001 and  # Allow small tolerance
                close[i] >= camarilla_S3_aligned[i] * 0.999 and
                ema_1d_aligned[i] > camarilla_S3_aligned[i] and  # Uptrend: EMA above S3
                volume_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches R3 in downtrend with volume confirmation
            elif (close[i] >= camarilla_R3_aligned[i] * 0.999 and  # Allow small tolerance
                  close[i] <= camarilla_R3_aligned[i] * 1.001 and
                  ema_1d_aligned[i] < camarilla_R3_aligned[i] and  # Downtrend: EMA below R3
                  volume_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals