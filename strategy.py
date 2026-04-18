#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter
Hypothesis: Trade Camarilla pivot breakouts at R1/S1 on 12h timeframe with volume confirmation and ATR volatility filter. Works in bull/bear markets by capturing breakouts from key intraday levels. Uses 1d Camarilla levels for structure and 12h for execution to avoid overtrading. Volume > 1.5x 24-period average confirms breakout strength. ATR filter avoids low-volatility false breakouts. Targets 15-25 trades/year via strict pivot breakout conditions + volume + volatility filters.
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    R1 = close_1d + (range_1d * 1.1 / 12)
    S1 = close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # ATR filter: avoid low volatility periods
    atr_period = 14
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full_like(close, np.nan)
    if n >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, n):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, atr_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume and volatility filters
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        vol_filter = atr[i] > np.nanpercentile(atr[start_idx:i+1], 20) if i > start_idx else True
        
        if position == 0:
            # Long: Close breaks above R1 + volume + volatility
            if close[i] > R1_aligned[i] and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 + volume + volatility
            elif close[i] < S1_aligned[i] and vol_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close breaks below S1 (reversal) or volatility drops
            if close[i] < S1_aligned[i] or atr[i] < np.nanpercentile(atr[max(0, i-20):i+1], 10):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close breaks above R1 (reversal) or volatility drops
            if close[i] > R1_aligned[i] or atr[i] < np.nanpercentile(atr[max(0, i-20):i+1], 10):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter"
timeframe = "12h"
leverage = 1.0