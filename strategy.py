#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + chop regime filter
# Donchian breakout provides clear structure-based entries with proven edge in crypto.
# Volume spike confirms institutional participation reducing false breakouts.
# Chop regime filter (CHOP > 61.8) avoids whipsaw in ranging markets.
# Designed for 12-35 trades/year (~50-140 total over 4 years) to minimize fee drag.
# Works in bull markets via breakout continuation and bear markets via breakdowns.
# ATR-based stoploss manages risk without intrabar simulation.

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
    
    # Get 1d data for volume and chop calculation - ONCE before loop
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
    
    # Calculate 1d Chopiness Index (CHOP) to detect ranging markets
    # CHOP = 100 * log10(sum(ATR(1)) / (n * ATR(n))) / log10(n)
    # We use 14-period CHOP as standard
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index 0
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).mean().values
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr1 / (14 * atr14)) / np.log10(14)
    chop_filter = chop > 61.8  # >61.8 = ranging market (mean revert), <38.2 = trending
    
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
        breakout_up = close[i] > highest_high[i-1]  # break above prior period high
        breakdown_down = close[i] < lowest_low[i-1]  # break below prior period low
        
        if position == 0:
            # Long: Donchian breakout up + volume spike + NOT choppy market (trending)
            if breakout_up and volume_spike_aligned[i] > 0.5 and chop_filter_aligned[i] < 0.5:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown down + volume spike + NOT choppy market (trending)
            elif breakdown_down and volume_spike_aligned[i] > 0.5 and chop_filter_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian breakdown or choppy market emerges
            if breakdown_down or chop_filter_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian breakout or choppy market emerges
            if breakout_up or chop_filter_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals