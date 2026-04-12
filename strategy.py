#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_Breakout_2025
Hypothesis: Combine daily Camarilla pivot levels with 12h trend filter and volume confirmation.
Long when price breaks above H4 with bullish 12h trend and volume spike.
Short when price breaks below L4 with bearish 12h trend and volume spike.
Uses tight entry conditions to limit trades (<50/year) and avoid fee drag.
Works in bull markets via breakout follow-through and in bear via mean reversion at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Pivot_Breakout_2025"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Camarilla: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    camarilla_h4 = pc + (ph - pl) * 1.1 / 2
    camarilla_l4 = pc - (ph - pl) * 1.1 / 2
    
    # Align to 4h (previous day's levels available at open)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 12h trend filter: EMA25
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema25 = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 25:
        alpha = 2 / (25 + 1)
        ema25[0] = close_12h[0]
        for i in range(1, len(close_12h)):
            ema25[i] = alpha * close_12h[i] + (1 - alpha) * ema25[i-1]
    
    ema25_aligned = align_htf_to_ltf(prices, df_12h, ema25)
    
    # Volume confirmation: 1.5x 20-period average
    if len(volume) >= 20:
        vol_avg = np.full(n, np.nan)
        for i in range(20, n):
            vol_avg[i] = np.mean(volume[i-20:i])
        vol_spike = volume > (vol_avg * 1.5)
    else:
        vol_spike = np.ones(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema25_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_h4_aligned[i]
        breakout_down = low[i] < camarilla_l4_aligned[i]
        
        # Trend filter
        trend_up = close[i] > ema25_aligned[i]
        trend_down = close[i] < ema25_aligned[i]
        
        # Entry logic
        long_entry = breakout_up and trend_up and vol_spike[i]
        short_entry = breakout_down and trend_down and vol_spike[i]
        
        # Exit logic: opposite breakout or trend reversal
        long_exit = (close[i] < camarilla_l4_aligned[i]) or (trend_down and not trend_up)
        short_exit = (close[i] > camarilla_h4_aligned[i]) or (trend_up and not trend_down)
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals