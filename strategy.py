#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeRegime
Hypothesis: Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (top 30%) works in both bull and bear markets. In bull: breakouts catch momentum. In bear: breakouts of lower Donchian band with price below EMA50 catch short squeezes or trend continuations. Volume ensures participation. Fixed size 0.25 limits trades to ~25/year.
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
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) channels on 4h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: volume > 70th percentile of 50-period lookback (high volume days only)
    vol_series = pd.Series(volume)
    vol_percentile_70 = vol_series.rolling(window=50, min_periods=50).quantile(0.70).values
    volume_regime = volume > vol_percentile_70
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for EMA and volume percentile, 20 for Donchian)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_percentile_70[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_channel = highest_high_20[i]
        lower_channel = lowest_low_20[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_regime = volume_regime[i]
        size = fixed_size
        
        # Entry conditions: Donchian breakout with volume regime AND 1d EMA50 trend filter
        long_entry = (close_val > upper_channel) and vol_regime and (close_val > ema_50_val)
        short_entry = (close_val < lower_channel) and vol_regime and (close_val < ema_50_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to Donchian midpoint
            mid_point = (upper_channel + lower_channel) / 2
            if close_val < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on mean reversion to Donchian midpoint
            mid_point = (upper_channel + lower_channel) / 2
            if close_val > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeRegime"
timeframe = "4h"
leverage = 1.0