#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyTrend_Filter_VolumeSpike
Hypothesis: 6h Donchian(20) breakouts filtered by 1w EMA200 trend direction and 1d volume spikes. Uses 0.25 position sizing to balance risk and return. Weekly trend ensures alignment with major market cycles, while Donchian breakouts capture momentum. Volume confirmation filters false breakouts. Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year). Works in bull markets (long when price > 1w EMA200) and bear markets (short when price < 1w EMA200).
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
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate Donchian channels (20-period) on 6h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1w EMA200 (200) + Donchian (20) + 1d volume avg (20)
    start_idx = max(200, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol = volume[i]
        ema_trend = ema_200_1w_aligned[i]
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        vol_avg = vol_avg_1d_aligned[i]
        
        # Volume spike: current 6h volume > 1.5 * 1d average volume (scaled for timeframe)
        # 1d volume represents ~4x 6h bars, so we adjust threshold
        volume_spike = vol > (1.5 * vol_avg / 4.0)
        
        if position == 0:
            # Long: price breaks above upper Donchian AND above 1w EMA200 AND volume spike
            if (close_val > upper_channel) and (close_val > ema_trend) and volume_spike:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian AND below 1w EMA200 AND volume spike
            elif (close_val < lower_channel) and (close_val < ema_trend) and volume_spike:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Long exit: price breaks below lower Donchian OR 1w EMA200 turns bearish
            if (close_val < lower_channel) or (close_val < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above upper Donchian OR 1w EMA200 turns bullish
            if (close_val > upper_channel) or (close_val > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_WeeklyTrend_Filter_VolumeSpike"
timeframe = "6h"
leverage = 1.0