#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and choppiness regime filter
# Long when: Price breaks above Donchian upper (20) AND 1d volume > 1.5 * 20-period average AND chop > 61.8 (range regime)
# Short when: Price breaks below Donchian lower (20) AND 1d volume > 1.5 * 20-period average AND chop > 61.8 (range regime)
# Exit when price returns to Donchian middle (mean of upper/lower)
# Donchian breakout captures volatility expansion after consolidation
# Volume spike confirms institutional participation
# Chop > 61.8 ensures we trade in ranging markets where breakouts are more reliable
# Works in both bull and bear markets by trading breakouts in direction of the squeeze break
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25

name = "4h_Donchian20_VolumeSpike_ChopRegime"
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
    
    # Get 1d data ONCE before loop for volume spike and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for calculations
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d True Range for choppy market indicator
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d choppy market indicator (Choppiness Index)
    # Chop = 100 * log10(sum(ATR14) / (log10(n) * (highest_high - lowest_low))) over period
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    log_n = np.log10(14)
    chop_1d = 100 * np.log10(sum_atr_14 / (log_n * (highest_high_14 - lowest_low_14)))
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 1d volume spike indicator
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_spike = volume_1d > (1.5 * vol_ma_20_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Donchian channels (20) on 4h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high_20
    donchian_lower = lowest_low_20
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(chop_1d_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: chop > 61.8 (range market) AND volume spike
        range_regime = chop_1d_aligned[i] > 61.8
        vol_spike = volume_spike_aligned[i] > 0.5  # Boolean converted to 0/1
        
        if position == 0:
            # Long: Break above upper Donchian in range regime with volume spike
            if close[i] > donchian_upper[i] and range_regime and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian in range regime with volume spike
            elif close[i] < donchian_lower[i] and range_regime and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to middle Donchian (mean reversion)
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to middle Donchian (mean reversion)
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals