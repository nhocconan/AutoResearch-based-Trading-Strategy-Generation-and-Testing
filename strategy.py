#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotTrend_1dVolumeSpike_v1
Hypothesis: On 6h timeframe, trade Donchian(20) breakouts aligned with weekly pivot trend direction and confirmed by 1d volume spike. Weekly pivot (R1/S1 from prior week) provides structural bias: long only when price > weekly R1, short only when price < weekly S1. Volume spike (1d volume > 1.5x 20-period average) confirms breakout strength. Designed for low frequency (12-30 trades/year) to avoid fee drag, works in bull/bear via weekly trend filter and volume confirmation.
"""

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
    
    # Get weekly data for pivot trend (R1/S1 from prior week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly Camarilla R1/S1 (pivot points from prior week)
    cam_high_1w = pd.Series(df_1w['high'].values).shift(1).values
    cam_low_1w = pd.Series(df_1w['low'].values).shift(1).values
    cam_close_1w = pd.Series(df_1w['close'].values).shift(1).values
    
    # Weekly R1, S1 levels
    weekly_R1 = cam_close_1w + (cam_high_1w - cam_low_1w) * 1.1 / 12
    weekly_S1 = cam_close_1w - (cam_high_1w - cam_low_1w) * 1.1 / 12
    
    # Get 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume average (20-period)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike_threshold = vol_ma_20 * 1.5  # 1.5x average volume
    
    # Get 6h data for Donchian(20) breakout
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    donch_high_20 = pd.Series(df_6h['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(df_6h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align all HTF indicators to 6h timeframe
    weekly_R1_aligned = align_htf_to_ltf(prices, df_1w, weekly_R1)
    weekly_S1_aligned = align_htf_to_ltf(prices, df_1w, weekly_S1)
    vol_spike_threshold_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_threshold)
    donch_high_20_aligned = align_htf_to_ltf(prices, df_6h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_6h, donch_low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of weekly pivot (need 2 bars for shift), 1d vol MA (20), 6h Donchian (20)
    start_idx = max(2, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_R1_aligned[i]) or 
            np.isnan(weekly_S1_aligned[i]) or
            np.isnan(vol_spike_threshold_aligned[i]) or
            np.isnan(donch_high_20_aligned[i]) or
            np.isnan(donch_low_20_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        weekly_r1 = weekly_R1_aligned[i]
        weekly_s1 = weekly_S1_aligned[i]
        vol_spike_thresh = vol_spike_threshold_aligned[i]
        donch_high = donch_high_20_aligned[i]
        donch_low = donch_low_20_aligned[i]
        
        if position == 0:
            # Long: break above Donchian HIGH with price > weekly R1 and volume spike
            long_signal = (high_val > donch_high) and \
                          (close_val > weekly_r1) and \
                          (volume_val > vol_spike_thresh)
            
            # Short: break below Donchian LOW with price < weekly S1 and volume spike
            short_signal = (low_val < donch_low) and \
                           (close_val < weekly_s1) and \
                           (volume_val > vol_spike_thresh)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below weekly S1 (trend reversal) or opposite Donchian break
            if close_val < weekly_s1 or low_val < donch_low:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above weekly R1 (trend reversal) or opposite Donchian break
            if close_val > weekly_r1 or high_val > donch_high:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotTrend_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0