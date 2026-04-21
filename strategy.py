#!/usr/bin/env python3
"""
6h_HTF_1d_WeeklyPivot_DonchianBreakout_VolumeFilter
Hypothesis: 6h Donchian(20) breakout in direction of weekly Camarilla pivot bias (price > weekly pivot = long bias, price < weekly pivot = short bias) with volume confirmation (>1.5x 20-bar volume MA). Weekly pivot provides structural bias from higher timeframe, reducing false breakouts in choppy markets. Works in bull via breakouts with long bias, in bear via breakdowns with short bias. Position size 0.25 balances risk/return. Target 12-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly Camarilla pivot levels ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point (PP) = (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly range
    range_1w = high_1w - low_1w
    # Camarilla levels: R4 = PP + range * 1.1/2, S4 = PP - range * 1.1/2
    # R3 = PP + range * 1.1/4, S3 = PP - range * 1.1/4
    # R1 = PP + range * 1.1/12, S1 = PP - range * 1.1/12
    # We use PP as bias filter, and R3/S3 for fade zones (not used here)
    # R4/S4 as breakout confirmation zones
    r4_1w = pp_1w + range_1w * 1.1 / 2.0
    s4_1w = pp_1w - range_1w * 1.1 / 2.0
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # === 6h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i])
            or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # Weekly pivot bias: price > PP = long bias, price < PP = short bias
        long_bias = price > pp_1w_aligned[i]
        short_bias = price < pp_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly R4 (strong breakout) + volume confirmation + long bias
            if price > highest_high[i-1] and price > r4_1w_aligned[i] and vol_ok and long_bias:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below weekly S4 (strong breakdown) + volume confirmation + short bias
            elif price < lowest_low[i-1] and price < s4_1w_aligned[i] and vol_ok and short_bias:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below Donchian low (reversal signal)
            if price < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above Donchian high (reversal signal)
            if price > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HTF_1d_WeeklyPivot_DonchianBreakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0