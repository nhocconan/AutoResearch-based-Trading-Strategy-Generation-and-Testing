#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeFilter
Hypothesis: On 6h timeframe, buy when price breaks above 20-period Donchian high with weekly pivot trend filter (price above weekly VWAP) and volume confirmation (above 70th percentile). Sell when price breaks below 20-period Donchian low with weekly pivot trend filter (price below weekly VWAP) and volume confirmation. Uses discrete position size 0.25 to limit trade frequency. Weekly VWAP provides structural trend filter that works in both bull (buying strength) and bear (selling weakness) markets. Volume ensures participation. Target: 12-25 trades/year per symbol.
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly VWAP (volume-weighted average price)
    typical_price_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    vol_1w = df_1w['volume'].values
    vwap_num = np.cumsum(typical_price_1w * vol_1w)
    vwap_den = np.cumsum(vol_1w)
    weekly_vwap = vwap_num / vwap_den
    weekly_vwap_aligned = align_htf_to_ltf(prices, df_1w, weekly_vwap)
    
    # Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 70th percentile of 50-period lookback
    vol_series = pd.Series(volume)
    vol_percentile_70 = vol_series.rolling(window=50, min_periods=50).quantile(0.70).values
    volume_filter = volume > vol_percentile_70
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for volume percentile, 20 for Donchian)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or 
            np.isnan(weekly_vwap_aligned[i]) or 
            np.isnan(vol_percentile_70[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donchian_high = highest_high_20[i]
        donchian_low = lowest_low_20[i]
        weekly_vwap_val = weekly_vwap_aligned[i]
        vol_filt = volume_filter[i]
        size = fixed_size
        
        # Entry conditions
        long_entry = (close_val > donchian_high) and (close_val > weekly_vwap_val) and vol_filt
        short_entry = (close_val < donchian_low) and (close_val < weekly_vwap_val) and vol_filt
        
        # Exit conditions: reverse signal when opposite Donchian break occurs with trend and volume
        long_exit = (close_val < donchian_low) and (close_val < weekly_vwap_val) and vol_filt
        short_exit = (close_val > donchian_high) and (close_val > weekly_vwap_val) and vol_filt
        
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
            # Long - exit on short entry signal
            if long_exit:
                signals[i] = -size
                position = -1
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on long entry signal
            if short_exit:
                signals[i] = size
                position = 1
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0