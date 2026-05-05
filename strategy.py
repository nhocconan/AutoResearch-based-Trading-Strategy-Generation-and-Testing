#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and chop regime filter
# Long when: Price breaks above 20-period 1w Donchian high AND 1w volume > 1.8x 20-period average AND chop > 61.8 (range regime on 1w)
# Short when: Price breaks below 20-period 1w Donchian low AND 1w volume > 1.8x 20-period average AND chop > 61.8 (range regime on 1w)
# Exit when price touches opposite Donchian level (20-period low for long exit, high for short exit)
# Uses 1w HTF for structure to avoid noise, 1d primary for timely execution
# Volume spike confirms institutional participation, chop filter ensures ranging markets where breakouts fail less
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25

name = "1d_Donchian20_1wVolumeSpike_Chop"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data ONCE before loop for Donchian levels, volume, and chop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need enough for calculations
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w True Range for chop
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1w Donchian channels (20-period)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w volume spike (current volume > 1.8x 20-period average)
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1w > (1.8 * vol_ma_20)
    
    # Calculate 1w Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(ATR14) / (n * (max(high) - min(low)))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, < 38.2 = trending
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Calculate sum of ATR14 over last 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    # Calculate max(high) - min(low) over last 14 periods
    max_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    # Avoid division by zero
    chop = 100 * (np.log10(sum_atr_14 / (14 * range_14)) / np.log10(14))
    # Handle invalid values
    chop = np.where((range_14 == 0) | np.isnan(chop), 50, chop)  # Default to neutral
    chop_regime = chop > 61.8  # Range regime
    
    # Align 1w indicators to 1d
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1w, vol_spike)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1w, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        vol_cond = bool(vol_spike_aligned[i])
        chop_cond = bool(chop_regime_aligned[i])
        
        if position == 0:
            # Long: Break above 1w Donchian high in range regime with volume spike
            if close[i] > donchian_high_aligned[i] and vol_cond and chop_cond:
                signals[i] = 0.25
                position = 1
            # Short: Break below 1w Donchian low in range regime with volume spike
            elif close[i] < donchian_low_aligned[i] and vol_cond and chop_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: touch 1w Donchian low (opposite level)
            if close[i] <= donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: touch 1w Donchian high (opposite level)
            if close[i] >= donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals