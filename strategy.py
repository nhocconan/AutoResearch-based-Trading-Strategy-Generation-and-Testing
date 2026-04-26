#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dRegime_VolumeFilter
Hypothesis: On 6h timeframe, use Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) from 1d timeframe to identify regime. Enter long when 1d Bull Power > 0 AND 6h price closes above 6h EMA20 AND 6h volume > 1.5x 20-period average volume. Enter short when 1d Bear Power < 0 AND 6h price closes below 6h EMA20 AND volume spike. Exit on opposite Elder Ray signal or EMA20 crossover. Uses 1d Elder Ray for trend regime and 6h EMA20/volume for timing, effective in both bull and bear markets via regime alignment. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = pd.Series(df_1d['close'].values)
    ema_13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align 1d Elder Ray to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 6h EMA20 for entry timing
    close_s = pd.Series(close)
    ema_20_6h = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 6h volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup and volume MA warmup
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(ema_20_6h[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d Elder Ray regime
        bull_regime = bull_power_1d_aligned[i] > 0
        bear_regime = bear_power_1d_aligned[i] < 0
        
        # 6h EMA20 filter
        price_above_ema = close[i] > ema_20_6h[i]
        price_below_ema = close[i] < ema_20_6h[i]
        
        if position == 0:
            # Long: 1d bull regime + 6h price above EMA20 + volume spike
            long_signal = bull_regime and price_above_ema and volume_spike[i]
            
            # Short: 1d bear regime + 6h price below EMA20 + volume spike
            short_signal = bear_regime and price_below_ema and volume_spike[i]
            
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
            # Exit: 1d bear regime OR price below EMA20
            if bear_regime or not price_above_ema:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: 1d bull regime OR price above EMA20
            if bull_regime or not price_below_ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dRegime_VolumeFilter"
timeframe = "6h"
leverage = 1.0