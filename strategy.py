#!/usr/bin/env python3
"""
12h_Camarilla_R4S4_Breakout_Volume_TrendFilter
Hypothesis: Use 12h primary timeframe with 1d Camarilla R4/S4 breakout for strong momentum capture.
Add volume confirmation (>2.0x 100-bar volume MA) and 1d EMA50 trend filter to reduce false breakouts.
Position size 0.25 balances risk/return. Target 12-37 trades/year per symbol.
Works in bull/bear via breakout logic and EMA filter reducing whipsaw in ranging markets.
Only long when price > 1d EMA50, only short when price < 1d EMA50.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R4, S4) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r4_1d = close_1d + (high_1d - low_1d) * 1.1 / 2.0  # R4 = Close + 1.1*(High-Low)/2
    s4_1d = close_1d - (high_1d - low_1d) * 1.1 / 2.0  # S4 = Close - 1.1*(High-Low)/2
    
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 1d EMA50 for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (100-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=100, min_periods=100).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if indicators not ready
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above 1d R4 + volume confirmation + price > 1d EMA50
            if price > r4_1d_aligned[i-1] and vol_ok and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d S4 + volume confirmation + price < 1d EMA50
            elif price < s4_1d_aligned[i-1] and vol_ok and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below 1d pivot (mean reversion) or trend changes
            pivot_1d = (high_1d + low_1d + close_1d) / 3.0
            pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
            if not np.isnan(pivot_1d_aligned[i]) and price < pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif price < ema_50_1d_aligned[i]:  # trend filter exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above 1d pivot (mean reversion) or trend changes
            pivot_1d = (high_1d + low_1d + close_1d) / 3.0
            pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
            if not np.isnan(pivot_1d_aligned[i]) and price > pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif price > ema_50_1d_aligned[i]:  # trend filter exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R4S4_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0