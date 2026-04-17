#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime
Strategy: 12h Camarilla pivot breakout with volume confirmation and chop regime filter.
Long: Price breaks above R1 + volume > 1.5x 20-period avg + chop < 61.8 (trending)
Short: Price breaks below S1 + volume > 1.5x 20-period avg + chop < 61.8 (trending)
Exit: Opposite breakout or price crosses daily VWAP
Position size: 0.25
Designed for trending markets with volume confirmation to avoid whipsaws.
Works in bull/bear by requiring trending regime (chop < 61.8).
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
    
    # Get daily data for Camarilla pivots and chop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate Chopiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # first period
    atr = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr * 14 / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 12h data for volume average
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    volume_ma20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(100, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume aligned to 12h
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        volume_filter = vol_12h_current > (1.5 * volume_ma20_12h_aligned[i])
        
        # Regime filter: chop < 61.8 indicates trending market
        trending_regime = chop_aligned[i] < 61.8
        
        # Camarilla breakout signals
        breakout_r1 = close[i] > r1_aligned[i]
        breakout_s1 = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: Break above R1 + volume + trending regime
            if breakout_r1 and volume_filter and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 + volume + trending regime
            elif breakout_s1 and volume_filter and trending_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Break below S1 or chop becomes too high (rangy)
            if breakout_s1 or chop_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Break above R1 or chop becomes too high (rangy)
            if breakout_r1 or chop_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0