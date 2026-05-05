#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and chop regime filter
# Long when: Price breaks above 20-period Donchian high AND 12h volume > 1.3x 20-period average AND chop < 38.2 (trending regime)
# Short when: Price breaks below 20-period Donchian low AND 12h volume > 1.3x 20-period average AND chop < 38.2 (trending regime)
# Exit when price touches opposite Donchian level (20-period low for long, high for short)
# Donchian provides clear trend structure with objective breakout levels
# Volume confirmation ensures institutional participation
# Chop regime filter (<38.2) ensures we trade in trending markets where breakouts work
# Target: 80-150 total trades over 4 years (20-37/year) with discrete sizing 0.25

name = "4h_Donchian20_12hVolume_TrendChop"
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
    
    # Get 12h data ONCE before loop for volume and chop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need enough for calculations
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h True Range for chop
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h Choppiness Index (CHOP)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Calculate sum of ATR14 over last 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    # Calculate max(high) - min(low) over last 14 periods
    max_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    # Avoid division by zero
    chop = 100 * (np.log10(sum_atr_14 / (14 * range_14)) / np.log10(14))
    # Handle invalid values
    chop = np.where((range_14 == 0) | np.isnan(chop), 50, chop)  # Default to neutral
    chop_regime = chop < 38.2  # Trending regime
    chop_regime_aligned = align_htf_to_ltf(prices, df_12h, chop_regime)
    
    # Calculate 12h volume spike (current volume > 1.3x 20-period average)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (1.3 * vol_ma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike)
    
    # Calculate 4h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        vol_cond = bool(vol_spike_aligned[i])
        chop_cond = bool(chop_regime_aligned[i])
        
        if position == 0:
            # Long: Break above Donchian high in trending regime with volume spike
            if close[i] > donch_high[i] and vol_cond and chop_cond:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low in trending regime with volume spike
            elif close[i] < donch_low[i] and vol_cond and chop_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: touch Donchian low
            if close[i] <= donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: touch Donchian high
            if close[i] >= donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals