#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and choppiness regime filter
# Uses Donchian channel breakouts for trend following, confirmed by 1d volume spikes
# and only trades when market is trending (CHOP < 38.2) to avoid whipsaw in ranging markets.
# Designed for 12-35 trades/year (~50-140 total over 4 years) to minimize fee drag.
# Works in both bull/bear markets by adapting to choppiness regime - trend following in trending markets.

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
    
    # Get 1d data for volume spike and choppiness calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR for choppiness indicator
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align length
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d True Range sum and ATR sum for choppiness
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    atr_sum = atr_1d * 14  # since ATR is already averaged
    
    # Choppiness Index: CHOP = 100 * log10(atr_sum / tr_sum) / log10(14)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    chop_raw = 100 * np.log10(atr_sum / tr_sum) / np.log10(14)
    chop_raw = np.nan_to_num(chop_raw, nan=50.0)  # neutral when undefined
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in trending markets (CHOP < 38.2)
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND volume spike AND trending market
            if (close[i] > donchian_high[i] and 
                volume_spike_aligned[i] > 0.5 and  # boolean as float
                is_trending):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND volume spike AND trending market
            elif (close[i] < donchian_low[i] and 
                  volume_spike_aligned[i] > 0.5 and
                  is_trending):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR chop becomes ranging
            if (close[i] < donchian_low[i]) or (chop_aligned[i] >= 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR chop becomes ranging
            if (close[i] > donchian_high[i]) or (chop_aligned[i] >= 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals