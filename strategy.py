#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and choppiness regime filter
# Uses Donchian channel breakouts for trend capture, confirmed by 1d volume spikes
# and filtered by 1d choppiness index to avoid whipsaw in ranging markets.
# Designed for 12-35 trades/year (~50-140 total over 4 years) to minimize fee drag.
# Works in both bull/bear markets by adapting to volatility regimes via choppiness filter.

name = "12h_Donchian20_1dVolumeSpike_ChopFilter_Trend"
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
    
    # Get 1d data for volume spike and choppiness filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14) for choppiness index
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.absolute(np.roll(close_1d, 1) - low_1d)))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d Choppiness Index (CHOP) = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    atr_sum_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.maximum(max_high_1d - min_low_1d, 1e-10)
    chop_1d = 100 * np.log10(atr_sum_1d / chop_denom) / np.log10(14)
    
    # Calculate 1d volume spike: volume > 2.0 * 20-period average volume
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Calculate 12h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when market is trending (CHOP < 38.2) or ranging (CHOP > 61.8) with volume spike
        is_trending = chop_1d_aligned[i] < 38.2
        is_ranging = chop_1d_aligned[i] > 61.8
        volume_confirmed = volume_spike_1d_aligned[i] > 0.5
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND volume spike AND (trending OR ranging with volume)
            if (close[i] > donchian_high[i] and 
                volume_confirmed and 
                (is_trending or is_ranging)):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND volume spike AND (trending OR ranging with volume)
            elif (close[i] < donchian_low[i] and 
                  volume_confirmed and 
                  (is_trending or is_ranging)):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR volume spike disappears
            if (close[i] <= donchian_high[i] and close[i] >= donchian_low[i]) or not volume_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR volume spike disappears
            if (close[i] <= donchian_high[i] and close[i] >= donchian_low[i]) or not volume_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals