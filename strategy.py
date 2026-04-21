#!/usr/bin/env python3
"""
6h_HTF_12h_1d_Camarilla_R4S4_Breakout_Volume_EMA34Filter
Hypothesis: Use 6h primary timeframe with 1d/12h Camarilla R4/S4 breakout for strong momentum capture.
Add volume confirmation (>2.0x 50-bar volume MA) and EMA34 filter on 6h to avoid false breakouts.
Position size 0.25 balances risk/return. Target 12-30 trades/year per symbol.
Breakouts work in bull/bear via momentum; EMA34 filter reduces whipsaw in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_1d) < 2 or len(df_12h) < 2:
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
    
    # === 12h EMA34 for trend filter ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 6h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (50-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(150, n):
        # Skip if indicators not ready
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) 
            or np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above 1d R4 + volume confirmation + price > 12h EMA34
            if price > r4_1d_aligned[i-1] and vol_ok and price > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d S4 + volume confirmation + price < 12h EMA34
            elif price < s4_1d_aligned[i-1] and vol_ok and price < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below 12h EMA34 (trend change) or volume dies
            if price < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above 12h EMA34 (trend change) or volume dies
            if price > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HTF_12h_1d_Camarilla_R4S4_Breakout_Volume_EMA34Filter"
timeframe = "6h"
leverage = 1.0