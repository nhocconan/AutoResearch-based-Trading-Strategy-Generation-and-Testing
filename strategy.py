#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Chop Index regime filter + 12h Donchian breakout + volume confirmation.
# Chop Index > 61.8 = ranging market (mean revert at Donchian bands)
# Chop Index < 38.2 = trending market (breakout follow)
# Long when Chop < 38.2 AND price breaks above 12h Donchian(20) upper band AND volume > 1.5x 20-period average
# Short when Chop < 38.2 AND price breaks below 12h Donchian(20) lower band AND volume > 1.5x 20-period average
# Exit when Chop > 61.8 (range) OR price reverts to Donchian midpoint
# Designed for 4h timeframe with low trade frequency (target: 20-50/year) to avoid fee drag.
# Uses Chop Index to avoid whipsaws in ranging markets and Donchian for clear breakout signals.
# Volume filter ensures participation and avoids low-conviction moves.
name = "4h_Chop_Donchian_Breakout_12hVol"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Chop Index (14-period)
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.where(
        (highest_high - lowest_low) > 0,
        100 * np.log10(atr14.sum() / (highest_high - lowest_low)) / np.log10(14),
        50
    )
    
    # 12h Donchian(20) channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donch_h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_l = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_m = (donch_h + donch_l) / 2
    
    donch_h_aligned = align_htf_to_ltf(prices, df_12h, donch_h)
    donch_l_aligned = align_htf_to_ltf(prices, df_12h, donch_l)
    donch_m_aligned = align_htf_to_ltf(prices, df_12h, donch_m)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(chop[i]) or np.isnan(donch_h_aligned[i]) or np.isnan(donch_l_aligned[i]) or 
            np.isnan(donch_m_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Chop < 38.2 (trending) AND price breaks above Donchian upper AND volume filter
            long_cond = (chop[i] < 38.2) and (close[i] > donch_h_aligned[i]) and volume_filter[i]
            # Short conditions: Chop < 38.2 (trending) AND price breaks below Donchian lower AND volume filter
            short_cond = (chop[i] < 38.2) and (close[i] < donch_l_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Chop > 61.8 (range) OR price returns to Donchian midpoint
            if chop[i] > 61.8 or close[i] < donch_m_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Chop > 61.8 (range) OR price returns to Donchian midpoint
            if chop[i] > 61.8 or close[i] > donch_m_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals