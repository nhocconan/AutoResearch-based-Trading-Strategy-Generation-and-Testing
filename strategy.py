#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + daily choppiness regime filter
# Long when: price breaks above 20-period 12h Donchian high AND 1d volume > 2x 20-period MA AND 1d chop > 61.8 (range) → mean reversion long at lower band
# Short when: price breaks below 20-period 12h Donchian low AND 1d volume > 2x 20-period MA AND 1d chop > 61.8 (range) → mean reversion short at upper band
# Uses Donchian for structure, volume for conviction, chop for regime (range = mean reversion)
# Timeframe: 12h, HTF: 1d for volume and chop. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_1dVolume_Chop_MeanReversion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 12h Donchian(20) - calculate on 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian channels
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe (already aligned via get_htf_data + align_htf_to_ltf)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # Get 1d data ONCE before loop for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d volume spike: > 2x 20-period MA
    if len(volume_1d) >= 20:
        vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume_1d > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(len(df_1d), dtype=bool)
    
    # 1d choppiness index: CHOP > 61.8 = range (mean reversion regime)
    if len(close_1d) >= 14:
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])
        
        # Sum of TR over 14 periods
        tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Chop = 100 * log10(sum(tr) / (hh - ll)) / log10(14)
        # Avoid division by zero
        range_hl = hh - ll
        chop = np.full_like(tr_sum, np.nan)
        valid = (range_ll > 0) & ~np.isnan(tr_sum) & ~np.isnan(range_hl)
        chop[valid] = 100 * np.log10(tr_sum[valid] / range_hl[valid]) / np.log10(14)
        
        chop_regime = chop > 61.8  # range = mean reversion
    else:
        chop_regime = np.zeros(len(df_1d), dtype=bool)
    
    # Align 1d indicators to 12h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume spike AND chop regime (range)
            if (close[i] > donch_high_aligned[i] and 
                volume_spike_aligned[i] == 1.0 and 
                chop_regime_aligned[i] == 1.0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume spike AND chop regime (range)
            elif (close[i] < donch_low_aligned[i] and 
                  volume_spike_aligned[i] == 1.0 and 
                  chop_regime_aligned[i] == 1.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR chop regime ends (trend starts)
            if (close[i] < donch_low_aligned[i] or chop_regime_aligned[i] == 0.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR chop regime ends (trend starts)
            if (close[i] > donch_high_aligned[i] or chop_regime_aligned[i] == 0.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals