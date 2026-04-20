#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with Volume Spike and Choppiness Filter
# - Entry: Price breaks above Donchian(20) high (long) or below Donchian(20) low (short)
# - Confirmation: Volume spike (current volume > 1.5 * 20-period average volume)
# - Filter: Choppiness Index > 61.8 (range) for mean reversion, < 38.2 (trend) for trend following
# - Exit: Opposite Donchian breakout or ATR-based stop (implemented via signal=0)
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for choppiness (using 14-period ATR)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])  # First TR is undefined
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate sum of true ranges for choppiness
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Calculate highest high and lowest low for choppiness
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum_tr_14 / (highest_high_14 - lowest_low_14)) / log10(14)
    # Avoid division by zero and log of zero/negative
    range_14 = highest_high_14 - lowest_low_14
    # Only calculate where range > 0 and sum_tr_14 > 0
    chop = np.full_like(close_1d, np.nan)
    mask = (range_14 > 0) & (sum_tr_14 > 0) & (~np.isnan(range_14)) & (~np.isnan(sum_tr_14))
    chop[mask] = 100 * np.log10(sum_tr_14[mask] / range_14[mask]) / np.log10(14)
    
    # Align choppiness to 4h timeframe
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian channels on 4h
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Donchian(20) channels
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(chop_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition
        vol_cond = vol_spike[i] if i < len(vol_spike) else False
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume spike + chop < 38.2 (trending)
            if (close_4h[i] > donch_high[i]) and vol_cond and (chop_4h[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + volume spike + chop < 38.2 (trending)
            elif (close_4h[i] < donch_low[i]) and vol_cond and (chop_4h[i] < 38.2):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR chop > 61.8 (ranging) for mean reversion
            if (close_4h[i] < donch_low[i]) or (chop_4h[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR chop > 61.8 (ranging) for mean reversion
            if (close_4h[i] > donch_high[i]) or (chop_4h[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0