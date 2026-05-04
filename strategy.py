#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and choppiness regime filter
# Uses 12h Donchian channels for structural breakouts, confirmed by 1d volume spikes (>2x 20-period average)
# and only trades when choppiness index (14) < 38.2 (trending regime) to avoid whipsaws in ranging markets.
# Designed for 12-35 trades/year (~50-140 total over 4 years) to minimize fee drag.
# Works in bull markets by capturing breakouts and in bear markets by filtering out false signals during consolidation.

name = "12h_Donchian20_1dVolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and choppiness - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    
    # Calculate 1d choppiness index: CHOP = 100 * log10(sum(atr(14)) / (max(high,14)-min(low,14))) / log10(14)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.absolute(np.roll(close_1d, 1) - low_1d)))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar TR
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high_14 - min_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop_value = 100 * np.log10(atr_14 * 14 / chop_denom) / np.log10(14)
    chop_value = np.where(np.isnan(chop_value), 50, chop_value)  # default to mid-range
    chop_trending = chop_value < 38.2  # trending regime
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending.astype(float))
    
    # Calculate 12h Donchian channels (20-period)
    max_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    min_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(max_high_20[i]) or np.isnan(min_low_20[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_trending_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        bull_breakout = close[i] > max_high_20[i-1]  # break above previous period's high
        bear_breakout = close[i] < min_low_20[i-1]   # break below previous period's low
        
        if position == 0:
            # Long: bullish breakout + volume spike + trending regime
            if (bull_breakout and 
                volume_spike_aligned[i] > 0.5 and  # boolean treated as 0/1
                chop_trending_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + volume spike + trending regime
            elif (bear_breakout and 
                  volume_spike_aligned[i] > 0.5 and 
                  chop_trending_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR volatility breaks down
            if (close[i] <= max_high_20[i] and close[i] >= min_low_20[i]) or volume_spike_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR volatility breaks down
            if (close[i] <= max_high_20[i] and close[i] >= min_low_20[i]) or volume_spike_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals