#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike filter and ATR stop
# - Long when price breaks above 20-period Donchian high on 12h with 1d volume > 1.5x 20-period average
# - Short when price breaks below 20-period Donchian low on 12h with 1d volume > 1.5x 20-period average
# - Donchian captures breakouts in both bull and bear markets
# - Volume filter ensures breakouts have conviction, reducing false signals
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on 1d
    avg_vol_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels on 12h
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 1d volume average to 12h timeframe
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(avg_vol_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol_1d_current = vol_1d[i // 12] if i >= 12 else 0  # 12h to 1d conversion (12 bars = 1 day)
        avg_vol = avg_vol_20_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high with volume spike
            if price > donchian_high[i] and vol_1d_current > 1.5 * avg_vol:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with volume spike
            elif price < donchian_low[i] and vol_1d_current > 1.5 * avg_vol:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or volume drops
            if price < donchian_low[i] or vol_1d_current < avg_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or volume drops
            if price > donchian_high[i] or vol_1d_current < avg_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike"
timeframe = "12h"
leverage = 1.0