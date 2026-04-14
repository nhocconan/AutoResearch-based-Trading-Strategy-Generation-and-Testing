#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour daily Camarilla pivot with volume confirmation and Choppiness regime filter.
# Camarilla levels derived from prior daily range provide statistically significant reversal zones.
# The Choppiness Index (14-period) filters for trending regimes (CHOP < 38.2) to trade with momentum.
# Volume > 1.5x 20-period average confirms institutional participation.
# This approach aims for 20-40 trades per year per symbol (80-160 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by combining mean-reversion at pivots with trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Camarilla pivot and Choppiness
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla pivot levels from previous day
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    camarilla_h5 = prev_close + 1.1 * range_ * 1.1 / 12
    camarilla_h4 = prev_close + 1.1 * range_ * 1.1 / 6
    camarilla_h3 = prev_close + 1.1 * range_ * 1.1 / 4
    camarilla_l3 = prev_close - 1.1 * range_ * 1.1 / 4
    camarilla_l4 = prev_close - 1.1 * range_ * 1.1 / 6
    camarilla_l5 = prev_close - 1.1 * range_ * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Choppiness Index (14-period) on daily
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).sum()
    
    # Choppiness: 100 * log10(sum(TR14)/(ATR14*14)) / log10(14)
    chop = 100 * np.log10(atr14 / (atr14))  # placeholder, will calculate properly
    # Recalculate correctly:
    sum_tr14 = tr.rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(sum_tr14 / (atr14 * 14)) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: trending market (Choppiness < 38.2)
        trending = chop_aligned[i] < 38.2
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long at L3 with trend filter and volume
            if (close[i] <= l3_aligned[i] and 
                trending and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short at H3 with trend filter and volume
            elif (close[i] >= h3_aligned[i] and 
                  trending and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches L4 or reverses at H3
            if close[i] >= l4_aligned[i] or close[i] >= h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches H4 or reverses at L3
            if close[i] <= h4_aligned[i] or close[i] <= l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Chop_Volume_v1"
timeframe = "4h"
leverage = 1.0