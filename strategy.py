#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian_Breakout_VolumeSpike_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian channels (20-period) on 6h
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume spike detection: current volume > 2.0 * 20-period average volume
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period - 1, n):
        vol_ma[i] = np.mean(volume[i - vol_period + 1:i + 1])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, vol_period, 34)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + price > daily EMA34 (uptrend)
            long_cond = (close[i] > highest_high[i]) and volume_spike[i] and (close[i] > ema34_1d_aligned[i])
            
            # Short: price breaks below Donchian low + volume spike + price < daily EMA34 (downtrend)
            short_cond = (close[i] < lowest_low[i]) and volume_spike[i] and (close[i] < ema34_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR trend changes
            if close[i] < lowest_low[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR trend changes
            if close[i] > highest_high[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian breakouts capture momentum, volume spikes confirm conviction, and daily EMA34 filter ensures
# trades align with higher timeframe trend. Works in bull markets (breakouts above EMA) and bear markets
# (breakouts below EMA). Target: 50-150 total trades over 4 years = 12-37/year to minimize fee decay.