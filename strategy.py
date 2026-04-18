#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Regime_v2
Hypothesis: Use 1d Camarilla pivot levels (R1/S1) for breakout direction, with volume confirmation and choppiness regime filter. 
Go long when price breaks above 1d R1 with volume > 1.5x average, short when price breaks below 1d S1 with volume > 1.5x average. 
Only trade when choppiness index (14) < 38.2 (trending regime). Uses 4h timeframe for reduced trade frequency. 
Target: 20-40 trades/year by combining pivot breakout with regime filter to avoid whipsaws in sideways markets.
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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    # Camarilla formulas: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    r1_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    
    if len(high_1d) >= 1:
        for i in range(len(high_1d)):
            if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
                r1_1d[i] = close_1d[i] + (high_1d[i] - low_1d[i]) * 1.1 / 12
                s1_1d[i] = close_1d[i] - (high_1d[i] - low_1d[i]) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Choppiness index (14) on 4h for regime filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    chop_period = 14
    atr_4h = np.full_like(close_4h, np.nan)
    chop_4h = np.full_like(close_4h, np.nan)
    
    if len(high_4h) >= chop_period:
        # Calculate True Range
        tr = np.full_like(close_4h, np.nan)
        for i in range(1, len(high_4h)):
            hl = high_4h[i] - low_4h[i]
            hc = np.abs(high_4h[i] - close_4h[i-1])
            lc = np.abs(low_4h[i] - close_4h[i-1])
            tr[i] = max(hl, hc, lc)
        
        # ATR(14)
        atr_4h[chop_period] = np.mean(tr[1:chop_period+1])
        for i in range(chop_period + 1, len(high_4h)):
            atr_4h[i] = (atr_4h[i-1] * (chop_period - 1) + tr[i]) / chop_period
        
        # Chop = 100 * log10(sum(TR14)/(n*ATR14)) / log10(n)
        for i in range(chop_period, len(high_4h)):
            if not np.isnan(atr_4h[i]) and atr_4h[i] > 0:
                sum_tr = np.sum(tr[i-chop_period+1:i+1])
                chop_4h[i] = 100 * np.log10(sum_tr / (chop_period * atr_4h[i])) / np.log10(chop_period)
    
    # Align chop to 4h timeframe (already on 4h, so just use directly)
    chop_4h_aligned = chop_4h  # Already on 4h timeframe
    
    # Volume confirmation: volume > 1.5x 20-period average on 4h
    vol_ma_4h = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma_4h[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, chop_period, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(chop_4h_aligned[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (chop < 38.2)
        trending_regime = chop_4h_aligned[i] < 38.2
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_4h[i]
        
        if position == 0 and trending_regime:
            # Long: price breaks above 1d R1 + volume
            if close[i] > r1_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d S1 + volume
            elif close[i] < s1_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 1d S1 (opposite level)
            if close[i] < s1_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 1d R1 (opposite level)
            if close[i] > r1_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Regime_v2"
timeframe = "4h"
leverage = 1.0