#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm_v1
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with weekly Camarilla pivot direction (R4/S4 from weekly pivot) and volume confirmation (>1.5x average) capture institutional moves. Weekly pivot direction provides bias: long when price above weekly R4, short when below weekly S4. This avoids counter-trend trades and works in bull/bear via weekly structure. Targets 12-37 trades/year on 6h with discrete sizing (0.25).
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
    
    # Load weekly data ONCE before loop for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from previous weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Camarilla R4 and S4 (strong breakout levels)
    weekly_range = high_1w - low_1w
    camarilla_r1w = close_1w + weekly_range * 1.1 / 12
    camarilla_s1w = close_1w - weekly_range * 1.1 / 12
    camarilla_r4w = close_1w + weekly_range * 1.1 / 2  # R4 = R3 + (R3-S3)*0.5, simplified to close + 1.1*range/2
    camarilla_s4w = close_1w - weekly_range * 1.1 / 2  # S4 = S3 - (R3-S3)*0.5
    
    # Align weekly levels to 6h (wait for completed weekly bar)
    camarilla_r4w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4w)
    camarilla_s4w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4w)
    
    # Donchian(20) channels on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Average volume for confirmation (24-period SMA = 4d on 6h)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Donchian(20), volume(24)
    start_idx = max(20, 24)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        r4w_val = camarilla_r4w_aligned[i]
        s4w_val = camarilla_s4w_aligned[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        
        # Skip if any data not ready
        if (np.isnan(r4w_val) or np.isnan(s4w_val) or np.isnan(avg_vol) or 
            np.isnan(donch_high) or np.isnan(donch_low)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Weekly pivot direction: price above weekly R4 = bullish bias, below S4 = bearish bias
        bullish_bias = close_val > r4w_val
        bearish_bias = close_val < s4w_val
        
        # Long: price breaks above Donchian high with bullish bias and volume
        long_condition = (high_val > donch_high) and bullish_bias and volume_confirmed
        # Short: price breaks below Donchian low with bearish bias and volume
        short_condition = (low_val < donch_low) and bearish_bias and volume_confirmed
        
        # Exit: price returns to mid-channel (Donchian midpoint)
        donchian_mid = (donch_high + donch_low) / 2
        long_exit = (position == 1 and close_val < donchian_mid)
        short_exit = (position == -1 and close_val > donchian_mid)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0