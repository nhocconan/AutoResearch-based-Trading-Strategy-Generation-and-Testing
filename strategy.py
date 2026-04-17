#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike and choppiness regime filter.
Long when price breaks above Camarilla R1 level with 1d volume > 2x 20-period average and CHOP > 61.8 (range).
Short when price breaks below Camarilla S1 level with 1d volume > 2x 20-period average and CHOP > 61.8.
Exit on opposite Camarilla level (S1 for long, R1 for short) or when CHOP < 38.2 (trend regime).
Uses discrete position sizing 0.25. Target: 75-200 total trades over 4 years.
Camarilla levels provide intraday support/resistance; volume confirms participation; chop filter avoids false breakouts in trends.
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
    
    # Get 1d data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (R1, S1)
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    def true_range(high_vals, low_vals, close_vals):
        tr1 = high_vals - low_vals
        tr2 = np.abs(high_vals - np.roll(close_vals, 1))
        tr3 = np.abs(low_vals - np.roll(close_vals, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first period
        return tr
    
    def choppiness_index(high_vals, low_vals, close_vals, window):
        tr = true_range(high_vals, low_vals, close_vals)
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        hh = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        chop = 100 * np.log10(atr.sum() / np.log(window) / (hh - ll)) / np.log10(window)
        # Handle division by zero and edge cases
        chop = np.where((hh - ll) == 0, 50, chop)
        chop = np.where(np.isnan(chop), 50, chop)
        return chop
    
    chop = choppiness_index(high_1d, low_1d, close_1d, 14)
    
    # Align all to primary timeframe (4h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for volume MA and CHOP
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2x 1d volume average (scaled)
        # Approximate 1d volume from 4h: 4h volume * 6 (since 6x 4h = 1d)
        volume_1d_estimate = volume[i] * 6
        volume_confirmed = volume_1d_estimate > 2.0 * vol_ma_20_1d_aligned[i]
        # Regime filter: CHOP > 61.8 = ranging market (good for mean reversion at pivots)
        ranging_market = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with volume and ranging market
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_confirmed and 
                ranging_market):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 with volume and ranging market
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_confirmed and 
                  ranging_market):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Camarilla S1 or chop < 38.2 (trend)
            if (close[i] < camarilla_s1_aligned[i] or 
                chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Camarilla R1 or chop < 38.2 (trend)
            if (close[i] > camarilla_r1_aligned[i] or 
                chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_Spike_Chop_Filter"
timeframe = "4h"
leverage = 1.0