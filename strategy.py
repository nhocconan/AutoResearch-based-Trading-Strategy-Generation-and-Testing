#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: 6h Donchian(20) breakouts with 1d EMA50 trend filter and volume spike >2x median capture medium-term momentum while avoiding whipsaws. Targets 12-30 trades/year per symbol. Works in bull (breakouts with trend) and bear (short breakdowns with downtrend). Discrete size 0.25 controls fee drag.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: volume > 2.0x 50-period median volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median_50 = volume_series.rolling(window=50, min_periods=50).median().values
    volume_spike = volume > (2.0 * vol_median_50)
    
    # 6h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Fixed position size
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 50 for 1d EMA, 50 for volume median, 20 for Donchian
    start_idx = max(50, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_median_50[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for breakout entry
            # Long: price breaks above upper Donchian with volume spike and uptrend (close > EMA50_1d)
            long_entry = (close_val > upper_channel) and vol_spike and (close_val > ema_50_val)
            # Short: price breaks below lower Donchian with volume spike and downtrend (close < EMA50_1d)
            short_entry = (close_val < lower_channel) and vol_spike and (close_val < ema_50_val)
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or price retouches lower channel
            if close_val < ema_50_val or close_val < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or price retouches upper channel
            if close_val > ema_50_val or close_val > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0