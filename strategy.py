#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: Trade 6h Donchian(20) breakouts with 12h EMA50 trend filter and volume spike confirmation.
Donchian channels provide objective breakout levels. 12h EMA50 ensures alignment with higher timeframe trend.
Volume spike (>2x 20-period median) confirms institutional interest. Designed for low trade frequency
(12-37/year) on 6h to minimize fee drag. Uses discrete position sizing (0.25) and works in both
bull/bear markets by following 12h trend. Only takes longs in uptrend, shorts in downtrend.
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
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian(20) channels: upper = max(high,20), lower = min(low,20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 2.0x median volume (20-period)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 12h EMA (50), Donchian (20), volume median (20)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_median[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_12h_val = ema_50_12h_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        if position == 0:
            # Long: break above Donchian upper with volume and uptrend (close > 12h EMA50)
            long_signal = (close_val > donchian_upper_aligned[i]) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          (close_val > ema_50_12h_val)
            
            # Short: break below Donchian lower with volume and downtrend (close < 12h EMA50)
            short_signal = (close_val < donchian_lower_aligned[i]) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           (close_val < ema_50_12h_val)
            
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
            # Exit: price breaks below Donchian lower (reversal) or trend changes (close < 12h EMA50)
            if (close_val < donchian_lower_aligned[i]) or \
               (close_val < ema_50_12h_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Donchian upper (reversal) or trend changes (close > 12h EMA50)
            if (close_val > donchian_upper_aligned[i]) or \
               (close_val > ema_50_12h_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0