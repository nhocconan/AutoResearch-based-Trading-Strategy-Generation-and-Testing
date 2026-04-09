#!/usr/bin/env python3
# 1d_1w_donchian_breakout_v1
# Hypothesis: 1-day breakout of weekly Donchian channels (20-period) with volume confirmation (>2x 20-day average volume).
# Weekly Donchian levels act as strong support/resistance; breaks signal momentum continuation.
# Volume filter reduces false breakouts. Works in bull markets (upward breaks) and bear markets (downward breaks).
# Target: 15-30 trades per year per symbol (~60-120 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate rolling max/min for Donchian channels
    donchian_high = np.full(len(weekly_high), np.nan)
    donchian_low = np.full(len(weekly_low), np.nan)
    
    for i in range(len(weekly_high)):
        if i >= 19:  # 20-period lookback
            donchian_high[i] = np.max(weekly_high[i-19:i+1])
            donchian_low[i] = np.min(weekly_low[i-19:i+1])
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation: 20-day average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below weekly Donchian low
            if close[i] <= donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above weekly Donchian high
            if close[i] >= donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly Donchian high with volume confirmation
            if close[i] > donchian_high_aligned[i] and volume[i] > vol_ma_20[i] * 2.0:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly Donchian low with volume confirmation
            elif close[i] < donchian_low_aligned[i] and volume[i] > vol_ma_20[i] * 2.0:
                position = -1
                signals[i] = -0.25
    
    return signals