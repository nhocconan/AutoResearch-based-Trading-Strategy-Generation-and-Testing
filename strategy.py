#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + choppiness regime filter
# Uses Donchian channel breakouts for trend capture, confirmed by 1d volume spikes (2x 20-day average)
# and choppiness regime (CHOP > 61.8 = range, CHOP < 38.2 = trend) to avoid false breakouts in chop.
# Designed for 20-50 trades/year (~80-200 total over 4 years) to minimize fee drag.
# Works in bull/bear markets: trend filter avoids whipsaw in ranging markets, volume confirms breakout strength.

name = "4h_Donchian20_1dVolumeSpike_ChopRegime"
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
    
    # Calculate 1d volume spike: current volume > 2x 20-day average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    
    # Calculate 1d choppiness index: CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low))
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / np.log10(highest_high - lowest_low))
    chop_regime = chop < 38.2  # trending regime (CHOP < 38.2)
    
    # Align 1d indicators to 4h timeframe (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime.astype(float))
    
    # Calculate 4h Donchian channels (20-period)
    highest_high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high_4h[i]) or np.isnan(lowest_low_4h[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band + volume spike + trending regime
            if (close[i] > highest_high_4h[i] and 
                volume_spike_aligned[i] > 0.5 and 
                chop_regime_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band + volume spike + trending regime
            elif (close[i] < lowest_low_4h[i] and 
                  volume_spike_aligned[i] > 0.5 and 
                  chop_regime_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR volume spike ends
            if (close[i] >= lowest_low_4h[i] and close[i] <= highest_high_4h[i]) or volume_spike_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR volume spike ends
            if (close[i] >= lowest_low_4h[i] and close[i] <= highest_high_4h[i]) or volume_spike_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals