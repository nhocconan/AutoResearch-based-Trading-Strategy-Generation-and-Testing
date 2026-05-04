#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike filter and chop regime
# Uses Donchian(20) channel breakouts for structure, confirmed by 1d volume spike (>1.5x 20-day avg volume)
# and choppiness regime filter (CHOP > 61.8 for mean reversion in ranging markets)
# Designed for 20-50 trades/year (~80-200 total over 4 years) to minimize fee drag
# Volume spike filters out low-momentum breakouts, chop filter avoids false signals in strong trends
# Works in bull/bear markets by combining breakout structure with regime-aware filtering

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
    
    # Get 1d data for volume spike filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d 20-period average volume for spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (vol_ma_20 * 1.5)
    
    # Align volume spike to 4h timeframe (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate choppiness index regime filter on 4h data
    atr_period = 14
    chop_period = 14
    tr1 = pd.Series(high).rolling(2).max() - pd.Series(low).rolling(2).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean().values
    
    sum_tr = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high_chop = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low_chop = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Avoid division by zero
    denominator = (highest_high_chop - lowest_low_chop)
    chop = np.where(denominator != 0, 
                    100 * np.log10(sum_tr / denominator) / np.log10(chop_period),
                    50)  # neutral value when range is zero
    
    # Chop regime: > 61.8 = ranging (mean revert), < 38.2 = trending (trend follow)
    chop_regime_ranging = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_regime_ranging[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND volume spike AND chop regime ranging
            if (close[i] > highest_high[i] and 
                volume_spike_aligned[i] > 0.5 and 
                chop_regime_ranging[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band AND volume spike AND chop regime ranging
            elif (close[i] < lowest_low[i] and 
                  volume_spike_aligned[i] > 0.5 and 
                  chop_regime_ranging[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR volume spike ends
            if (close[i] <= highest_high[i] and close[i] >= lowest_low[i]) or volume_spike_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR volume spike ends
            if (close[i] <= highest_high[i] and close[i] >= lowest_low[i]) or volume_spike_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals