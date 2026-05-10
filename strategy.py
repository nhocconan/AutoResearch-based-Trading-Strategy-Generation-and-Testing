#!/usr/bin/env python3
# 6H_Donchian_20_WeeklyTrend_1dVolumeSpike
# Hypothesis: Buy Donchian(20) breakouts in direction of weekly trend with daily volume confirmation.
# Weekly trend filter ensures we trade only in the dominant long-term direction (works in bull/bear).
# Daily volume spike filters for institutional participation, reducing false breakouts.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).

name = "6H_Donchian_20_WeeklyTrend_1dVolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h indicators
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Donchian channels (20-period)
    donch_high = high_s.rolling(window=20, min_periods=20).max().values
    donch_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend direction
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly trend to 6h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Daily volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (vol_ma_1d * 2.0)  # 2x average daily volume
    
    # Align daily volume spike to 6h
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        vol_spike = volume_spike_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: weekly uptrend + price breaks above Donchian high + volume spike
            if weekly_up and close[i] > donch_high[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + price breaks below Donchian low + volume spike
            elif weekly_down and close[i] < donch_low[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: weekly trend weakens or price returns to Donchian mid-point
            donch_mid = (donch_high[i] + donch_low[i]) / 2
            if not weekly_up or close[i] < donch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: weekly trend weakens or price returns to Donchian mid-point
            donch_mid = (donch_high[i] + donch_low[i]) / 2
            if not weekly_down or close[i] > donch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals