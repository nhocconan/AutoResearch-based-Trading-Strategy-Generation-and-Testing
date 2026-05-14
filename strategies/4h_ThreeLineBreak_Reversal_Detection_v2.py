#!/usr/bin/env python3
"""
4h_ThreeLineBreak_Reversal_Detection_v2
Hypothesis: Use Three Line Break (TLB) reversal patterns on 4h combined with volume confirmation and 1w EMA trend filter to capture medium-term reversals in both bull and bear markets. TLB filters out minor fluctuations and highlights significant trend changes. Volume confirms institutional participation. Weekly EMA ensures alignment with higher timeframe trend. Designed for low trade frequency (<30/year) to minimize fee drag while maintaining edge.
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
    
    # Calculate Three Line Break (TLB) on close prices
    # TLB: new line up if close > prior high, new line down if close < prior low
    # We'll track the last three closing prices for simplicity
    tl_up = np.zeros(n, dtype=bool)
    tl_down = np.zeros(n, dtype=bool)
    
    # Initialize
    last_close = close[0]
    last_high = high[0]
    last_low = low[0]
    
    for i in range(1, n):
        if close[i] > last_high:
            tl_up[i] = True
            last_high = close[i]
            last_low = close[i]  # reset on new up line
        elif close[i] < last_low:
            tl_down[i] = True
            last_low = close[i]
            last_high = close[i]  # reset on new down line
        else:
            # No new line, carry forward levels
            last_high = max(last_high, close[i])
            last_low = min(last_low, close[i])
        last_close = close[i]
    
    # 1w EMA trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: >1.8x 20-period average (moderate threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need some history for TLB
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_1w_val = ema_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: TLB up with volume and above weekly EMA
            if tl_up[i] and vol_spike and price > ema_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: TLB down with volume and below weekly EMA
            elif tl_down[i] and vol_spike and price < ema_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: TLB down reversal or below weekly EMA
            if tl_down[i] or price < ema_1w_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: TLB up reversal or above weekly EMA
            if tl_up[i] or price > ema_1w_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_ThreeLineBreak_Reversal_Detection_v2"
timeframe = "4h"
leverage = 1.0