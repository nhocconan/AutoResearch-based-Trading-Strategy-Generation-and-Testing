#!/usr/bin/env python3
"""
1d_1W_Camarilla_R4_S4_Breakout_With_Volume_Filter
Hypothesis: Use weekly Camarilla pivot levels to identify breakout points on daily chart.
Go long when price breaks above weekly S4 with volume confirmation, short when breaks below weekly R4.
Weekly structure provides stronger support/resistance, reducing false breakouts.
Volume confirms institutional participation. Designed for low-frequency, high-conviction trades.
Target: 10-20 trades/year with position size 0.25 to minimize fee drag.
Works in bull/bear markets by capturing strong momentum moves after consolidation.
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
    
    # Get weekly data for stronger structural levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    # R4 = Close + 1.5 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    camarilla_r4 = close_1w + 1.5 * (high_1w - low_1w)
    camarilla_s4 = close_1w - 1.5 * (high_1w - low_1w)
    
    # Align weekly levels to daily timeframe (wait for weekly bar close)
    r4_1d = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    s4_1d = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Volume confirmation: 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_1d[i]) or np.isnan(s4_1d[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-day average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above weekly S4 with volume confirmation
            if close[i] > s4_1d[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly R4 with volume confirmation
            elif close[i] < r4_1d[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses back below weekly S4
            if close[i] < s4_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above weekly R4
            if close[i] > r4_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_Camarilla_R4_S4_Breakout_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0