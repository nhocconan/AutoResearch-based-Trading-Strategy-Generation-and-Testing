#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot bias and volume confirmation.
# Uses weekly pivot levels (PP, R1, S1) from 1w data for bias, daily Donchian channel for entry,
# and volume spike for confirmation. Long when price breaks above Donchian high in bullish weekly bias
# (close > weekly PP), short when breaks below Donchian low in bearish bias (close < weekly PP).
# Designed for 6h timeframe to target 12-30 trades/year per symbol.
# Weekly pivot bias provides structural bias that works in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for weekly pivot bias (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    # PP = (high + low + close) / 3
    # R1 = 2*PP - low
    # S1 = 2*PP - high
    pp = (high_1w + low_1w + close_1w) / 3
    r1 = 2 * pp - low_1w
    s1 = 2 * pp - high_1w
    
    # Align weekly pivot to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Load 1d data for Donchian channel (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channel (20-period high/low)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + bullish weekly bias (close > PP) + volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > pp_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + bearish weekly bias (close < PP) + volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < pp_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on Donchian low touch or bearish bias shift
                if (close[i] < donchian_low_aligned[i] or close[i] < pp_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Donchian high touch or bullish bias shift
                if (close[i] > donchian_high_aligned[i] or close[i] > pp_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Bias_VolumeSpike"
timeframe = "6h"
leverage = 1.0