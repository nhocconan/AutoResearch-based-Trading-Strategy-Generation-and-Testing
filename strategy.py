#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout with 1d Volume Spike and Chop Regime Filter
# Long when: Price breaks above R3 (1d) AND 1d volume > 1.5x 20-period average AND chop > 61.8 (range regime)
# Short when: Price breaks below S3 (1d) AND 1d volume > 1.5x 20-period average AND chop > 61.8 (range regime)
# Exit when price touches R1/S1 (1d) or opposite Camarilla level
# Camarilla levels from 1d provide institutional support/resistance
# Volume spike confirms institutional participation
# Chop regime filter (>61.8) ensures we trade in ranging markets where mean reversion works
# Target: 100-180 total trades over 4 years (25-45/year) with discrete sizing 0.25

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_Chop"
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
    
    # Get 1d data ONCE before loop for Camarilla levels, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for calculations
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d True Range for chop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d CAMARILLA levels (based on previous day)
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # But we use current day's high/low for breakout (standard Camarilla breakout)
    # Actually, Camarilla uses previous day's range to calculate today's levels
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Calculate Camarilla levels from previous day's data
    R3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d)
    S3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d)
    R1 = prev_close_1d + 0.5 * (prev_high_1d - prev_low_1d)
    S1 = prev_close_1d - 0.5 * (prev_high_1d - prev_low_1d)
    
    # Align Camarilla levels to 4h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Calculate 1d volume spike (current volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * vol_ma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Calculate 1d Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(ATR14) / (n * (max(high) - min(low)))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, < 38.2 = trending
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Calculate sum of ATR14 over last 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    # Calculate max(high) - min(low) over last 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    # Avoid division by zero
    chop = 100 * (np.log10(sum_atr_14 / (14 * range_14)) / np.log10(14))
    # Handle invalid values
    chop = np.where((range_14 == 0) | np.isnan(chop), 50, chop)  # Default to neutral
    chop_regime = chop > 61.8  # Range regime
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        vol_cond = bool(vol_spike_aligned[i])
        chop_cond = bool(chop_regime_aligned[i])
        
        if position == 0:
            # Long: Break above R3 in range regime with volume spike
            if close[i] > R3_aligned[i] and vol_cond and chop_cond:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 in range regime with volume spike
            elif close[i] < S3_aligned[i] and vol_cond and chop_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: touch R1 or break below S3 (reversal)
            if close[i] <= R1_aligned[i] or close[i] < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: touch S1 or break above R3 (reversal)
            if close[i] >= S1_aligned[i] or close[i] > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals