#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with Volume Confirmation and Chop Filter
# Uses Donchian(20) breakout for directional entries, volume spike for confirmation,
# and Choppiness Index(14) > 61.8 for range filter to avoid whipsaw.
# Designed to capture breakouts in trending markets while avoiding range-bound periods.
# Target: 50-150 total trades over 4 years (12-38/year).

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    # Calculate Choppiness Index on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(tr_sum / (atr_1d * 14)) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for Donchian calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Range filter: only trade when market is trending (Choppiness < 61.8)
        if chop_aligned[i] >= 61.8:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high with volume spike
            if close[i] > donchian_high[i] and vol_spike[i]:
                position = 1
                signals[i] = position_size
            # Short entry: price breaks below Donchian low with volume spike
            elif close[i] < donchian_low[i] and vol_spike[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_Volume_Chop_Filter"
timeframe = "4h"
leverage = 1.0