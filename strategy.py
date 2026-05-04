#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + choppiness regime filter
# Uses 12h Donchian channel breakouts for trend capture, confirmed by 1d volume spikes
# and filtered by 1d choppiness regime (CHOP > 61.8 = range, avoid breakouts in chop).
# Designed for 12-35 trades/year (~50-140 total over 4 years) to minimize fee drag.
# Works in bull/bear markets: breakouts capture trends, chop filter avoids whipsaws in ranging markets.

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
    
    # Get 1d data for volume spike and choppiness - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike: volume > 2.0 * 20-period SMA
    vol_sma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_sma_20)
    
    # Calculate 1d choppiness index: CHOP(14) = 100 * log10(sum(TR(14)) / (HHV(14,high) - LLV(14,low))) / log10(14)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr1])  # align with index
    
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    chop_regime = chop > 61.8  # >61.8 = ranging market (avoid breakouts)
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime.astype(float))
    
    # Calculate 12h Donchian channels: 20-period high/low
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike and chop regime as confirmation filters
        vol_spike = volume_spike_aligned[i] > 0.5
        in_chop = chop_regime_aligned[i] > 0.5  # True = ranging market
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume spike AND NOT in choppy regime
            if (close[i] > donch_high[i] and vol_spike and not in_chop):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume spike AND NOT in choppy regime
            elif (close[i] < donch_low[i] and vol_spike and not in_chop):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR volume spike fades
            if (close[i] <= donch_high[i] and close[i] >= donch_low[i]) or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR volume spike fades
            if (close[i] <= donch_high[i] and close[i] >= donch_low[i]) or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals