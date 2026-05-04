#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + choppiness regime filter
# Uses Donchian channel breakouts for trend capture, confirmed by 1d volume spikes
# and filtered by choppiness index to avoid whipsaw in ranging markets.
# Designed for 20-50 trades/year (~80-200 total over 4 years) to minimize fee drag.
# Works in bull markets via breakouts and in bear markets via short breakdowns.
# Choppiness filter ensures we only trade when market is trending (CHOP < 38.2) or
# mean-reverting (CHOP > 61.8) depending on Donchian position.

name = "4h_Donchian20_1dVolumeSpike_ChoppinessFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike and choppiness - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    
    # Calculate 1d choppiness index: CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low))
    # Simplified: CHOP = 100 * log10(sum(tr) / log10(highest_high - lowest_low)) / log10(14)
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first TR
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = highest_high_14 - lowest_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # avoid div by zero
    
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    chopiness_1d = 100 * np.log10(sum_tr_14) / np.log10(chop_denominator) / np.log10(14)
    
    # Align 1d indicators to 4h timeframe (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chopiness_1d_aligned = align_htf_to_ltf(prices, df_1d, chopiness_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chopiness_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high_20[i-1]  # break above previous period high
        breakout_down = close[i] < lowest_low_20[i-1]   # break below previous period low
        
        # Choppiness regime: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
        chop_value = chopiness_1d_aligned[i]
        is_trending = chop_value < 38.2
        is_ranging = chop_value > 61.8
        
        if position == 0:
            # Long: Donchian breakout up + volume spike + trending market
            if (breakout_up and 
                volume_spike_aligned[i] > 0.5 and  # aligned as float, threshold at 0.5
                is_trending):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + volume spike + trending market
            elif (breakout_down and 
                  volume_spike_aligned[i] > 0.5 and
                  is_trending):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian breakdown OR chop indicates ranging market
            if (close[i] < lowest_low_20[i] or is_ranging):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian breakout up OR chop indicates ranging market
            if (close[i] > highest_high_20[i] or is_ranging):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals