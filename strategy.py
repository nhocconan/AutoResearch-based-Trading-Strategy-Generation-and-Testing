# 12h_Donchian20_1wVolumeSpike_TrendFilter
# Hypothesis: Use 1-week Donchian channel breakout with 1-day volume spike confirmation
# on 12-hour timeframe to capture major trend moves while avoiding false breakouts.
# Donchian provides clear trend structure, volume confirms institutional interest,
# and weekly timeframe filters noise. Designed for 15-30 trades/year to minimize fee drag.
# Works in bull markets (breakouts up) and bear markets (breakouts down).

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
    
    # === Weekly Donchian Channel (20 periods) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian channels
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # === Daily Volume Spike (2.0x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for weekly calculations
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-day average
        vol_today_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_today_aligned[i] > vol_ma_20_1d_aligned[i] * 2.0
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume confirmation
            if close[i] > donchian_high_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly Donchian low with volume confirmation
            elif close[i] < donchian_low_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price returns to opposite Donchian level
        elif position == 1:
            # Exit long: price crosses below weekly Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wVolumeSpike_TrendFilter"
timeframe = "12h"
leverage = 1.0