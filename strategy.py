#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1w Camarilla pivot (R1/S1) breakout + volume confirmation + 1d chop regime filter.
Long when price breaks above 1w Camarilla R1 with volume > 1.5x 20-period volume average and 1d chop < 61.8 (trending regime).
Short when price breaks below 1w Camarilla S1 with volume > 1.5x 20-period volume average and 1d chop < 61.8 (trending regime).
Camarilla pivots from weekly timeframe provide strong institutional levels; volume confirms breakout authenticity;
chop filter ensures we only trade in trending markets to avoid whipsaws in ranging conditions.
Designed to work in both bull and bear markets by trading breakouts in the direction of the trend.
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
    
    # Get 1w data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivots (R1, S1)
    def camarilla_pivots(high_vals, low_vals, close_vals):
        typical = (high_vals + low_vals + close_vals) / 3.0
        range_val = high_vals - low_vals
        R1 = close_vals + range_val * 1.1 / 12.0
        S1 = close_vals - range_val * 1.1 / 12.0
        return R1, S1
    
    camarilla_R1_1w, camarilla_S1_1w = camarilla_pivots(high_1w, low_1w, close_1w)
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Chop Index (14-period)
    def chop_index(high_vals, low_vals, close_vals, window):
        atr_sum = np.zeros_like(close_vals)
        true_range = np.maximum(high_vals - low_vals, 
                               np.maximum(np.abs(high_vals - np.roll(close_vals, 1)), 
                                          np.abs(low_vals - np.roll(close_vals, 1))))
        true_range[0] = high_vals[0] - low_vals[0]  # first bar TR
        atr_sum = pd.Series(true_range).rolling(window=window, min_periods=1).sum().values
        highest_high = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(window)
        # Handle division by zero or invalid values
        chop = np.where((highest_high - lowest_low) == 0, 50.0, chop)
        chop = np.where(np.isnan(chop), 50.0, chop)
        return chop
    
    chop_14_1d = chop_index(high_1d, low_1d, close_1d, 14)
    
    # Calculate 12h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w Camarilla pivots to 12h timeframe
    camarilla_R1_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R1_1w)
    camarilla_S1_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S1_1w)
    
    # Align 1d chop index to 12h timeframe
    chop_14_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_14_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # need enough for volume MA and indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_R1_1w_aligned[i]) or 
            np.isnan(camarilla_S1_1w_aligned[i]) or 
            np.isnan(chop_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime filter: only trade when chop < 61.8 (trending market)
        trending_regime = chop_14_1d_aligned[i] < 61.8
        
        if position == 0:
            # Long: price breaks above 1w Camarilla R1 with volume and trending regime
            if (close[i] > camarilla_R1_1w_aligned[i] and 
                volume_confirmed and 
                trending_regime):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Camarilla S1 with volume and trending regime
            elif (close[i] < camarilla_S1_1w_aligned[i] and 
                  volume_confirmed and 
                  trending_regime):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 1w Camarilla S1 (opposite side)
            if close[i] < camarilla_S1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 1w Camarilla R1 (opposite side)
            if close[i] > camarilla_R1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1wCamarilla_R1S1_Breakout_Volume_ChopFilter"
timeframe = "12h"
leverage = 1.0