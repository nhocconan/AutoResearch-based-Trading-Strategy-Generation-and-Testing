#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Camarilla R1/S1 breakout + volume spike + chop regime filter.
Long when price breaks above R1 with volume > 2.0x 20-period average and CHOP > 61.8 (range regime).
Short when price breaks below S1 with volume > 2.0x 20-period average and CHOP > 61.8.
Uses discrete position sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
Camarilla levels provide intraday support/resistance; volume confirms participation; chop filter ensures ranging market for mean reversion.
Designed to work in ranging markets (chop high) where price respects Camarilla levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and chop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla R1 and S1 levels
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Calculate 1d Choppiness Index (CHOP)
    def true_range(high_vals, low_vals, close_vals):
        tr1 = high_vals - low_vals
        tr2 = np.abs(high_vals - np.roll(close_vals, 1))
        tr3 = np.abs(low_vals - np.roll(close_vals, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high_vals[0] - low_vals[0]  # first period
        return tr
    
    def choppiness_index(high_vals, low_vals, close_vals, window):
        tr = true_range(high_vals, low_vals, close_vals)
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        hh = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        chop = 100 * np.log10(atr.sum() / (hh - ll)) / np.log10(window)
        return chop
    
    chop_14 = choppiness_index(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (12h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    chop_14_aligned = align_htf_to_ltf(prices, df_1d, chop_14)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for CAM and CHOP
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(chop_14_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x 20-day average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20_1d_aligned[i]
        # Chop filter: CHOP > 61.8 indicates ranging market (mean reversion regime)
        ranging_regime = chop_14_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above R1 with volume and ranging regime
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_confirmed and 
                ranging_regime):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and ranging regime
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_confirmed and 
                  ranging_regime):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below S1 (opposite Camarilla level)
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above R1 (opposite Camarilla level)
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dCamarilla_R1S1_Volume_Spike_Chop_Filter"
timeframe = "12h"
leverage = 1.0