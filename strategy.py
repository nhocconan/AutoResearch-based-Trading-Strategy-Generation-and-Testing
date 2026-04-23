#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 12h Volume Spike and Choppiness Filter
- Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) identifies trend: 
  Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish)
- 12h Volume Spike (>2.0x 20-period average) confirms momentum behind breakout
- 4h Choppiness Index (CHOP) > 61.8 = ranging market (avoid entries), < 38.2 = trending (allow entries)
- Designed for 4h timeframe to capture medium-term swings with low frequency
- Uses proven Alligator trend + volume + regime filters to minimize false signals
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Williams Alligator (SMAs)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Calculate 4h Choppiness Index
    chop_period = 14
    atr_series = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))))
    atr_series.iloc[0] = high[0] - low[0]  # first ATR
    atr_sum = atr_series.rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    # Avoid division by zero
    range_val = highest_high - lowest_low
    range_val = np.where(range_val == 0, 1e-10, range_val)
    chop = 100 * np.log10(atr_sum / range_val) / np.log10(chop_period)
    
    # Calculate 12h Volume Spike filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(jaw_period, teeth_period, lips_period, chop_period, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator bullish alignment AND volume spike AND trending market (CHOP < 38.2)
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                volume[i] > 2.0 * vol_ma_12h_aligned[i] and 
                chop[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish alignment AND volume spike AND trending market (CHOP < 38.2)
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  volume[i] > 2.0 * vol_ma_12h_aligned[i] and 
                  chop[i] < 38.2):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator reverses alignment OR chop becomes too high (ranging market)
            exit_signal = False
            
            if position == 1:
                # Exit long when Alligator turns bearish OR chop > 61.8 (ranging)
                if lips[i] < teeth[i] or chop[i] > 61.8:
                    exit_signal = True
            elif position == -1:
                # Exit short when Alligator turns bullish OR chop > 61.8 (ranging)
                if lips[i] > teeth[i] or chop[i] > 61.8:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_VolumeSpike_ChoppinessFilter"
timeframe = "4h"
leverage = 1.0