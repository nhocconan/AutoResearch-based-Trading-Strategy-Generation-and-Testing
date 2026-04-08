#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_spike_v2
# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume spike confirmation and trend filter.
# Camarilla levels identify key support/resistance zones where price reverses or breaks.
# Volume spike confirms institutional participation in breakouts.
# Trend filter (weekly EMA) ensures we trade with the higher timeframe trend.
# Designed to work in both bull and bear markets by capturing reversals at key levels and breakouts with volume.
# Target: 15-30 trades/year with ~0.25 position size to minimize fee drag.

name = "12h_camarilla_pivot_volume_spike_v2"
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
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using typical Camarilla formula based on previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot and levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    # Resistance levels
    r1 = pivot + (range_hl * 1.1 / 12)
    r2 = pivot + (range_hl * 1.1 / 6)
    r3 = pivot + (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    # Support levels
    s1 = pivot - (range_hl * 1.1 / 12)
    s2 = pivot - (range_hl * 1.1 / 6)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Align to 12h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Weekly trend filter (EMA 50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume spike: current volume > 2x 24-period average
    vol_period = 24
    vol_ma = np.zeros_like(volume)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    vol_ma[:vol_period-1] = vol_ma[vol_period-1]
    
    # Start from sufficient lookback
    start_idx = max(vol_period, 2) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price reaches S3 or weekly trend turns bearish
            if close[i] <= s3_aligned[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price reaches R3 or weekly trend turns bullish
            if close[i] >= r3_aligned[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches S1 with volume spike and above weekly EMA
            if (abs(close[i] - s1_aligned[i]) / s1_aligned[i] < 0.005) and volume_filter and close[i] > ema_50_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price touches R1 with volume spike and below weekly EMA
            elif (abs(close[i] - r1_aligned[i]) / r1_aligned[i] < 0.005) and volume_filter and close[i] < ema_50_aligned[i]:
                position = -1
                signals[i] = -0.25
            # Long breakout: price breaks above R1 with volume spike
            elif close[i] > r1_aligned[i] and volume_filter and close[i] > ema_50_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below S1 with volume spike
            elif close[i] < s1_aligned[i] and volume_filter and close[i] < ema_50_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals