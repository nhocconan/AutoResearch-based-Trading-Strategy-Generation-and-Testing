#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1_S1_Breakout_Volume_Regime
Strategy: 12-hour breakout of daily Camarilla R1/S1 levels with volume confirmation and chop regime filter.
Long: Price breaks above daily R1 + volume > 1.5x 20-period average + CHOP > 61.8 (range)
Short: Price breaks below daily S1 + volume > 1.5x 20-period average + CHOP > 61.8
Exit: Price returns to opposite Camarilla level (S1 for long, R1 for short)
Position size: 0.25
Designed to capture mean-reversion bounces in ranging markets with volume confirmation.
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
    
    # Calculate Camarilla levels from previous 1-day OHLC
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Camarilla formula: range = high - low
    # R1 = close + (range * 1.1/12)
    # S1 = close - (range * 1.1/12)
    range_1d = high_1d - low_1d
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Chopiness Index (14-period) for regime filter - range when > 61.8
    # CHOP = 100 * log10(sum(ATR14) / (max(HH14) - min(LL14))) / log10(14)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # first period
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * (np.log10(atr14 * 14) / np.log10(14)) / np.log10((hh14 - ll14) + 1e-10)
    chop = np.where((hh14 - ll14) > 0, chop, 50)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Regime filter: choppy market (CHOP > 61.8) for mean reversion
        regime_filter = chop[i] > 61.8
        
        # Breakout conditions
        breakout_above_r1 = close[i] > r1_1d_aligned[i-1]
        breakout_below_s1 = close[i] < s1_1d_aligned[i-1]
        
        # Exit conditions: return to opposite level
        return_to_s1 = abs(close[i] - s1_1d_aligned[i]) < 0.001 * close[i]  # within 0.1% of S1
        return_to_r1 = abs(close[i] - r1_1d_aligned[i]) < 0.001 * close[i]  # within 0.1% of R1
        
        if position == 0:
            # Long: break above R1 + volume + chop regime
            if breakout_above_r1 and volume_filter and regime_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 + volume + chop regime
            elif breakout_below_s1 and volume_filter and regime_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to S1 or break above R1 again
            if return_to_s1 or breakout_above_r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to R1 or break below S1 again
            if return_to_r1 or breakout_below_s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_R1_S1_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0