#!/usr/bin/env python3
"""
12h_1d_1w_camarilla_volume_v1
Hypothesis: Camarilla pivot levels from 1d with volume confirmation and 1w trend filter.
- Entry: Price touches Camarilla S3 (long) or R3 (short) with volume > 1.5x 20-period average
- Trend filter: 1w EMA(50) direction (only long if 1w uptrend, short if downtrend)
- Exit: Opposite Camarilla level touch (S1 for long, R1 for short) or trend reversal
- Position sizing: 0.25 for long, -0.25 for short
- Target: 12-37 trades/year (50-150 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_camarilla_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # S4 = close - 1.5*(high-low), S3 = close - 1.1*(high-low), etc.
    range_1d = high_1d - low_1d
    camarilla_s3 = close_1d - 1.1 * range_1d
    camarilla_s1 = close_1d - 1.05 * range_1d
    camarilla_r1 = close_1d + 1.05 * range_1d
    camarilla_r3 = close_1d + 1.1 * range_1d
    
    # Forward fill the levels (each level is valid until next 1d bar)
    camarilla_s3_series = pd.Series(camarilla_s3)
    camarilla_s1_series = pd.Series(camarilla_s1)
    camarilla_r1_series = pd.Series(camarilla_r1)
    camarilla_r3_series = pd.Series(camarilla_r3)
    
    camarilla_s3_ffilled = camarilla_s3_series.ffill().values
    camarilla_s1_ffilled = camarilla_s1_series.ffill().values
    camarilla_r1_ffilled = camarilla_r1_series.ffill().values
    camarilla_r3_ffilled = camarilla_r3_series.ffill().values
    
    # Align 1d Camarilla levels to 12h
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_ffilled)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_ffilled)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_ffilled)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_ffilled)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema_50_1w
    trend_1w_down = close_1w < ema_50_1w
    
    # Forward fill trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().values
    
    # Align 1w trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # Volume filter: 12h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price touches S1 OR 1w trend turns down
            if low[i] <= camarilla_s1_aligned[i] or trend_1w_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price touches R1 OR 1w trend turns up
            if high[i] >= camarilla_r1_aligned[i] or trend_1w_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Price touches S3 + 1w uptrend + volume
            if low[i] <= camarilla_s3_aligned[i] and trend_1w_up_aligned[i] and volume_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches R3 + 1w downtrend + volume
            elif high[i] >= camarilla_r3_aligned[i] and trend_1w_down_aligned[i] and volume_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals