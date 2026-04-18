#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Regime_v1
Hypothesis: Use daily Camarilla pivot levels (R1/S1) for breakout entries with volume confirmation and choppiness regime filter. 
Go long when price breaks above daily R1 with volume > 1.5x average and chop > 61.8 (ranging market). 
Go short when price breaks below daily S1 with volume > 1.5x average and chop > 61.8. 
Exit when price returns to daily pivot (PP) or chop < 38.2 (trending market). 
Designed for choppy/range markets (2025-2026) with tight stops to avoid whipsaw. 
Target: 20-40 trades/year by requiring confluence of breakout, volume, and regime.
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels
    R1 = np.full_like(close_1d, np.nan)
    S1 = np.full_like(close_1d, np.nan)
    PP = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 1:
        for i in range(len(close_1d)):
            # Camarilla formulas
            PP[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
            range_ = high_1d[i] - low_1d[i]
            R1[i] = close_1d[i] + range_ * 1.1 / 12
            S1[i] = close_1d[i] - range_ * 1.1 / 12
    
    # Align daily Camarilla levels to 4h timeframe
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    PP_4h = align_htf_to_ltf(prices, df_1d, PP)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Choppiness index (14-period) for regime filter
    chop = np.full_like(close, np.nan)
    chop_period = 14
    
    if len(high) >= chop_period and len(low) >= chop_period and len(close) >= chop_period:
        for i in range(chop_period, len(close)):
            # True range
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr = max(tr1, tr2, tr3)
            
            # Sum of true ranges
            atr_sum = 0
            for j in range(i - chop_period + 1, i + 1):
                tr1_j = high[j] - low[j]
                tr2_j = abs(high[j] - close[j-1])
                tr3_j = abs(low[j] - close[j-1])
                tr_j = max(tr1_j, tr2_j, tr3_j)
                atr_sum += tr_j
            
            if atr_sum > 0:
                chop[i] = 100 * np.log10(chop_period / atr_sum) / np.log10(2)
            else:
                chop[i] = 50  # neutral
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, chop_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or np.isnan(PP_4h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter: chop > 61.8 = ranging (good for mean reversion at pivots)
        ranging = chop[i] > 61.8
        
        if position == 0 and ranging:
            # Long: price breaks above R1 with volume
            if close[i] > R1_4h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume
            elif close[i] < S1_4h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to PP or chop < 38.2 (trending)
            if close[i] < PP_4h[i] or chop[i] < 38.2:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to PP or chop < 38.2 (trending)
            if close[i] > PP_4h[i] or chop[i] < 38.2:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0