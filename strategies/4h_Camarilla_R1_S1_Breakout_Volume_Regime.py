#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Regime
Hypothesis: Use daily Camarilla R1/S1 levels with volume confirmation and a chop regime filter to capture breakouts with follow-through. Works in bull and bear markets by avoiding choppy conditions and only trading when price breaks key daily support/resistance with volume. Targets 20-40 trades/year with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R1 = Close + 1.1 * (High - Low)
    # S1 = Close - 1.1 * (High - Low)
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (wait for daily bar close)
    r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Calculate chop regime filter using 14-period high-low range vs true range
    # Chop > 61.8 = ranging (avoid), Chop < 38.2 = trending (favor)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr14 = np.full(n, np.nan)
    for i in range(14, n):
        atr14[i] = np.mean(tr[i-13:i+1])  # simple moving average of TR
    hl14 = np.full(n, np.nan)
    for i in range(14, n):
        hl14[i] = np.max(high[i-13:i+1]) - np.min(low[i-13:i+1])
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if atr14[i] > 0:
            chop[i] = 100 * np.log10(hl14[i] / (atr14[i] * 14)) / np.log10(14)
        else:
            chop[i] = 50  # neutral if ATR is zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # need volume MA and chop
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter: only trade when chop < 50 (trending bias)
        trending_regime = chop[i] < 50
        
        if position == 0:
            # Long entry: price breaks above S1 with volume confirmation and trending regime
            if close[i] > s1_4h[i] and vol_confirmed and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below R1 with volume confirmation and trending regime
            elif close[i] < r1_4h[i] and vol_confirmed and trending_regime:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses back below S1
            if close[i] < s1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above R1
            if close[i] > r1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0