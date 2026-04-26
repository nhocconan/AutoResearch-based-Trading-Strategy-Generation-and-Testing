#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: Donchian(20) breakouts on 4h with 1d EMA50 trend filter and volume spike (top 20%). Only trade breakouts aligned with daily trend to avoid counter-trend whipsaws. Volume spike ensures institutional participation. Fixed size 0.25 to limit trades (~20-30/year). Works in bull/bear via trend filter + volume confirmation.
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
    
    # Previous 1d bar's OHLC for Donchian levels (20-period lookback on daily)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Donchian channels: 20-period high/low on daily timeframe
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d_vals)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: volume > 80th percentile of 50-period lookback (high volume days only)
    vol_series = pd.Series(volume)
    vol_percentile_80 = vol_series.rolling(window=50, min_periods=50).quantile(0.80).values
    volume_spike = volume > vol_percentile_80
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for EMA and Donchian/volume percentile)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_percentile_80[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        # Entry conditions: breakout of Donchian bands with volume spike AND aligned with 1d EMA50 trend
        long_entry = (close_val > donchian_high_val) and vol_spike and (close_val > ema_50_val)
        short_entry = (close_val < donchian_low_val) and vol_spike and (close_val < ema_50_val)
        
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
            # Long - exit on mean reversion to midpoint (Donchian center)
            mid_point = (donchian_high_val + donchian_low_val) / 2
            if close_val < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on mean reversion to midpoint (Donchian center)
            mid_point = (donchian_high_val + donchian_low_val) / 2
            if close_val > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0