#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Elder Ray (Bull/Bear Power) + 6h Donchian(20) breakout.
Long when 1d Bull Power > 0 (buying pressure) and price breaks above 6h Donchian(20) high with volume confirmation.
Short when 1d Bear Power < 0 (selling pressure) and price breaks below 6h Donchian(20) low with volume confirmation.
Elder Ray measures daily buying/selling power relative to EMA13, providing regime-specific bias.
Donchian breakout captures momentum, volume confirms strength. Works in bull (buy power + upward breakout)
and bear (sell power + downward breakout) markets by requiring alignment between daily bias and 6h breakout.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d_s = pd.Series(close_1d)
    ema13_1d = close_1d_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray components
    bull_power_1d = high_1d - ema13_1d  # Buying power: high minus EMA
    bear_power_1d = low_1d - ema13_1d   # Selling power: low minus EMA
    
    # Calculate 6h Donchian(20) channels
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Elder Ray components to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # need enough for EMA13 and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: daily buying pressure + 6h Donchian breakout above + volume
            if (bull_power_1d_aligned[i] > 0 and 
                close[i] > donchian_high[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: daily selling pressure + 6h Donchian breakdown below + volume
            elif (bear_power_1d_aligned[i] < 0 and 
                  close[i] < donchian_low[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 6h Donchian low (mean reversion)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 6h Donchian high (mean reversion)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dElderRay_Donchian20_Breakout_Volume_Confirm"
timeframe = "6h"
leverage = 1.0