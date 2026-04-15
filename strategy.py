#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with Volume Spike and Daily Choppiness Filter
# Uses previous day's Camarilla levels (H3/L3) as key support/resistance.
# Long: Break above H3 with volume spike in choppy market (CHOP > 61.8).
# Short: Break below L3 with volume spike in choppy market.
# Works in bull/bear by fading extremes in ranging markets.
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla levels and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's Camarilla levels (H3, L3)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for previous day
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Shift by 1 to avoid look-ahead (use previous day's levels)
    prev_h3 = np.roll(camarilla_h3, 1)
    prev_l3 = np.roll(camarilla_l3, 1)
    prev_h3[0] = np.nan
    prev_l3[0] = np.nan
    
    # Choppiness Index (14-period) on 1d
    # CHOP = 100 * log10(sum(TR)/ (ATR * n)) / log10(n)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr / (atr_1d * 14 + 1e-10)) / np.log10(14)
    chop[np.isnan(chop)] = 50  # Default to middle when not enough data
    
    # Align to 12h timeframe
    prev_h3_aligned = align_htf_to_ltf(prices, df_1d, prev_h3)
    prev_l3_aligned = align_htf_to_ltf(prices, df_1d, prev_l3)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(prev_h3_aligned[i]) or np.isnan(prev_l3_aligned[i]) or
            np.isnan(chop_aligned[i])):
            continue
        
        # Volume spike: current volume > 2x median of last 20 periods
        vol_median = np.median(volume[max(0, i-20):i+1])
        volume_spike = volume[i] > 2.0 * vol_median
        
        # Long: price breaks above H3 + volume spike + choppy market (CHOP > 61.8)
        if (close[i] > prev_h3_aligned[i] and
            volume_spike and
            chop_aligned[i] > 61.8 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: price breaks below L3 + volume spike + choppy market (CHOP > 61.8)
        elif (close[i] < prev_l3_aligned[i] and
              volume_spike and
              chop_aligned[i] > 61.8 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or market becomes trending (CHOP < 38.2)
        elif position == 1 and (close[i] < prev_l3_aligned[i] or chop_aligned[i] < 38.2):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > prev_h3_aligned[i] or chop_aligned[i] < 38.2):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_Volume_CHOP"
timeframe = "12h"
leverage = 1.0