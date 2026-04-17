#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_VolumeSpike_V1
Trend-following on 1d using weekly Donchian breakout with volume confirmation.
Long when price > weekly Donchian high + volume > 1.5x 20d avg.
Short when price < weekly Donchian low + volume > 1.5x 20d avg.
Exit when price crosses weekly Donchian midline or volume drops.
Position size: 0.30. Target: 10-20 trades/year.
Works in bull/bear: captures breakouts in trending markets, volume filter avoids false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian high (20-period)
    donch_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Weekly Donchian low (20-period)
    donch_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    # Weekly Donchian midline
    donch_mid_1w = (donch_high_1w + donch_low_1w) / 2
    
    # Align weekly Donchian levels to daily
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high_1w)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low_1w)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1w, donch_mid_1w)
    
    # Daily volume filter
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # warmup for weekly Donchian and volume MA
        # Skip if any required data is not available
        if np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or np.isnan(donch_mid_aligned[i]) or np.isnan(volume_ma20[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long when price breaks above weekly Donchian high + volume spike
            if close[i] > donch_high_aligned[i] and volume_filter:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below weekly Donchian low + volume spike
            elif close[i] < donch_low_aligned[i] and volume_filter:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly Donchian midline OR volume drops
            if close[i] < donch_mid_aligned[i] or volume[i] < (1.2 * volume_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price crosses above weekly Donchian midline OR volume drops
            if close[i] > donch_mid_aligned[i] or volume[i] < (1.2 * volume_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_WeeklyDonchian_Breakout_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0