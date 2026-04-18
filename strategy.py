#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime_v1
12h strategy using Camarilla pivot levels (R1/S1) from daily with volume confirmation and choppiness regime filter.
- Long: Close breaks above R1 + volume > 1.3x daily avg + chop > 61.8 (range)
- Short: Close breaks below S1 + volume > 1.3x daily avg + chop > 61.8 (range)
- Exit: Opposite breakout or chop < 38.2 (trend)
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in ranging markets (mean reversion at extremes) and avoids trending markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and volatility
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels (R1, S1) from previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    rang = high_1d - low_1d
    r1 = close_1d + 1.1 * rang / 12
    s1 = close_1d - 1.1 * rang / 12
    
    # Align Camarilla levels to 12h (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Choppiness Index (14-period) on daily
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr1 = np.concatenate([[high_1d[0] - low_1d[0]], tr1])
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr1.sum() / (max_high - min_low)) / np.log10(14) if (max_high - min_low) > 0 else 50
    # Fix: calculate properly per bar
    chop_series = []
    for i in range(len(high_1d)):
        if i < 13:
            chop_series.append(50)
        else:
            tr_sum = pd.Series(tr1[max(0, i-13):i+1]).sum()
            hh = high_1d[max(0, i-13):i+1].max()
            ll = low_1d[max(0, i-13):i+1].min()
            if hh - ll > 0:
                chop_val = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
                chop_series.append(chop_val)
            else:
                chop_series.append(50)
    chop_values = np.array(chop_series)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need enough for chop calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma_aligned[i]
        
        # Regime filter: only trade in ranging markets (chop > 61.8)
        in_range = chop_aligned[i] > 61.8
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakdown_down = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: range + volume + breakout above R1
            if in_range and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: range + volume + breakdown below S1
            elif in_range and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakout below S1 or regime shifts to trend
            if breakdown_down or chop_aligned[i] < 38.2:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above R1 or regime shifts to trend
            if breakout_up or chop_aligned[i] < 38.2:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0