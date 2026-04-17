#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Camarilla R1/S1 breakout + volume confirmation + choppiness regime filter.
Long when price breaks above 1d Camarilla R1 level with volume > 1.5x 20-period average and choppy market (CHOP > 61.8).
Short when price breaks below 1d Camarilla S1 level with volume > 1.5x 20-period average and choppy market (CHOP > 61.8).
Camarilla levels from daily timeframe provide institutional support/resistance, volume confirms institutional participation,
and choppiness filter ensures we only trade in ranging markets where mean reversion at these levels works best.
Designed to work in ranging markets (2025 bearish bias) while avoiding strong trends where breakouts fail.
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
    
    # Get 1d data for Camarilla levels and chop filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (R1, S1)
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Calculate 1d Choppiness Index (CHOP)
    def choppiness_index(high_vals, low_vals, close_vals, window):
        atr_sum = pd.Series(np.maximum(np.maximum(high_vals - low_vals, np.abs(high_vals - np.roll(close_vals, 1))), 
                                       np.maximum(np.abs(low_vals - np.roll(close_vals, 1)), np.abs(np.roll(close_vals, 1) - np.roll(close_vals, 2)))),
                            window=window, min_periods=1).sum().values
        highest_high = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(window)
        # Handle division by zero and invalid values
        chop = np.where((highest_high - lowest_low) == 0, 50, chop)
        chop = np.where(np.isnan(chop) | np.isinf(chop), 50, chop)
        return chop
    
    chop_14_1d = choppiness_index(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (12h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    chop_14_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_14_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(chop_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # Choppiness filter: choppy market (CHOP > 61.8) for mean reversion
        chop_filter = chop_14_1d_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with volume and choppy market
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_confirmed and 
                chop_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 with volume and choppy market
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_confirmed and 
                  chop_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Camarilla S1 (opposite level)
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Camarilla R1 (opposite level)
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dCamarilla_R1S1_Breakout_Volume_ChopFilter"
timeframe = "12h"
leverage = 1.0