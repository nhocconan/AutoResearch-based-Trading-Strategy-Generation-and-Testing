#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + chop regime filter
# Uses 12h Donchian channel breakouts for trend capture, confirmed by 1d volume spikes
# and filtered by 1d choppiness index to avoid whipsaw in ranging markets.
# Designed for 12-35 trades/year (~50-140 total over 4 years) to minimize fee drag.
# Works in both bull/bear markets: breakouts capture trends, chop filter avoids false signals in ranges.

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
    
    # Get 1d data for volume spike and chop filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    
    # Calculate 1d choppiness index: CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(14)
    # Using ATR(1) = True Range for simplicity
    tr1 = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    atr1_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1_sum / (max_high - min_low + 1e-10)) / np.log10(14)
    chop_filter = chop > 61.8  # ranging market (mean reversion regime)
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter.astype(float))
    
    # Calculate 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # break above previous high
        breakout_down = close[i] < lowest_low[i-1]  # break below previous low
        
        if position == 0:
            # Long: Donchian breakout up + volume spike + NOT in chop regime (trending market)
            if (breakout_up and 
                volume_spike_aligned[i] > 0.5 and 
                chop_filter_aligned[i] < 0.5):  # not choppy = trending
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + volume spike + NOT in chop regime
            elif (breakout_down and 
                  volume_spike_aligned[i] > 0.5 and 
                  chop_filter_aligned[i] < 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian breakdown or chop regime emerges
            if (close[i] < lowest_low[i] or chop_filter_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian breakout up or chop regime emerges
            if (close[i] > highest_high[i] or chop_filter_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals