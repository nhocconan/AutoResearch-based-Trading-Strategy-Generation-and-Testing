#!/usr/bin/env python3
"""
1h Volume Spike + 4h Donchian Breakout
Hypothesis: Intraday volume spikes combined with 4h Donchian breakouts capture momentum with filtered entries.
Uses 4h Donchian channels for trend direction and 1h volume spikes for entry timing, reducing false breakouts.
Designed for 15-30 trades/year on 1h timeframe to avoid fee drag. Works in bull/bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Donchian channels (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1h volume spike: 2.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        
        if position == 0:
            # Long: break above 4h Donchian high with volume spike
            if price > upper and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below 4h Donchian low with volume spike
            elif price < lower and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.20
            # Exit: price returns to 4h Donchian low
            if price <= lower:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.20
            # Exit: price returns to 4h Donchian high
            if price >= upper:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_VolumeSpike_DonchianBreakout_4h"
timeframe = "1h"
leverage = 1.0