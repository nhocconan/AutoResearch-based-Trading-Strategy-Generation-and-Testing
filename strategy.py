#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and choppiness regime filter
# Long when: Price breaks above Donchian upper channel (20) AND 1d volume > 1.5x 20-period average AND CHOP(14) > 61.8 (rangy market)
# Short when: Price breaks below Donchian lower channel (20) AND 1d volume > 1.5x 20-period average AND CHOP(14) > 61.8
# Exit when price returns to Donchian middle (mean reversion in ranging markets)
# Donchian breakout captures volatility expansion after consolidation
# Volume spike confirms institutional participation
# Choppiness filter ensures we trade breakouts from ranging conditions (avoid trending markets where breakouts fail)
# Works in both bull and bear markets by trading range breakouts
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "12h_DonchianBreakout_VolumeChop"
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
    
    # Get 1d data ONCE before loop for volume spike and choppiness filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for calculations
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d True Range for ATR (used in Choppiness)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan  # First value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Choppiness Index: CHOP = 100 * log10(sum(ATR14)/log(n)) / log10(n)
    # Simplified: CHOP = 100 * log10(rolling_sum(ATR14, 14) / (ATR * 14)) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr_14 / (atr_1d * 14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1d volume spike: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Donchian Channels (20) on 12h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_20 = (highest_20 + lowest_20) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(middle_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime: rangy market (CHOP > 61.8) AND volume spike
        rangy_market = chop_aligned[i] > 61.8
        vol_confirm = volume_spike_aligned[i] > 0.5  # Boolean as float
        
        if position == 0:
            # Long: Break above upper Donchian in rangy market with volume
            if close[i] > highest_20[i] and rangy_market and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian in rangy market with volume
            elif close[i] < lowest_20[i] and rangy_market and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to middle Donchian (mean reversion)
            if close[i] < middle_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to middle Donchian (mean reversion)
            if close[i] > middle_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals