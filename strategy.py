#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_Volume_Trend_HTF4h_1d
Hypothesis: 1h price breaks above/below daily Camarilla R1/S1 with volume spike and 4h EMA trend filter.
Uses 4h EMA for trend direction (more responsive than 1d) and 1d pivot for key levels.
Targets 15-30 trades/year to avoid fee drag while capturing strong directional moves.
Works in bull (breakouts) and bear (breakdowns) via trend filter.
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
    
    # Daily Camarilla pivot from previous 1d bar
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot levels: P = (H+L+C)/3, R1 = C + (H-L)*1.1/2, S1 = C - (H-L)*1.1/2
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align to 1h: previous day's levels available after 1d bar closes
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Trend filter: 4h EMA34 (more responsive than 1d)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_4h_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema34 = ema_34_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and uptrend
            if price > r1_val and vol_spike and price > ema34:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume spike and downtrend
            elif price < s1_val and vol_spike and price < ema34:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            signals[i] = 0.20
            # Exit: price closes below pivot OR trend turns down
            if price < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.20
            # Exit: price closes above pivot OR trend turns up
            if price > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_Volume_Trend_HTF4h_1d"
timeframe = "1h"
leverage = 1.0