#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1S1_Breakout_Volume_Regime
Strategy: 12-hour Camarilla pivot level breakout with volume confirmation and 1d chop regime filter.
Long: Price breaks above Camarilla R1 + volume > 1.5x average + 1d CHOP > 61.8 (range)
Short: Price breaks below Camarilla S1 + volume > 1.5x average + 1d CHOP > 61.8 (range)
Exit: Price returns to Camarilla pivot point (PP)
Position size: 0.25
Designed to fade false breakouts in ranging markets while capturing true breakouts with volume.
Timeframe: 12h
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
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # Using previous day's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First value will be invalid due to roll, but we'll handle with start_idx
    
    PP = (prev_high + prev_low + prev_close) / 3.0
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12.0
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12.0
    
    # Calculate 1-day CHOP (choppiness index) for regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # First TR will be invalid due to roll, but we'll handle with start_idx
    
    # ATR(14) for 1d
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # CHOP = 100 * log15(sum(ATR(14)) / (max(high) - min(low)) over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14)) / np.log10(15)
    
    # Align 1d CHOP to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20)  # Need enough data for CHOP ATR and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(PP[i]) or 
            np.isnan(R1[i]) or 
            np.isnan(S1[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Regime filter: 1d CHOP > 61.8 indicates ranging market (good for fade)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        # Breakout conditions
        breakout_up = close[i] > R1[i]  # break above R1
        breakout_down = close[i] < S1[i]  # break below S1
        
        # Return to pivot point (PP)
        return_to_pp = abs(close[i] - PP[i]) < 0.005 * PP[i]  # within 0.5% of PP
        
        if position == 0:
            # In ranging markets, fade false breakouts
            # Long: price breaks below S1 but volume confirms and we expect reversion to PP
            # Short: price breaks above R1 but volume confirms and we expect reversion to PP
            if breakout_down and volume_filter and chop_filter:
                # Price broke below S1, expect reversion to PP (long)
                signals[i] = 0.25
                position = 1
            elif breakout_up and volume_filter and chop_filter:
                # Price broke above R1, expect reversion to PP (short)
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to PP or break above R1 (invalidates the fade)
            if return_to_pp or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to PP or break below S1 (invalidates the fade)
            if return_to_pp or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_R1S1_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0