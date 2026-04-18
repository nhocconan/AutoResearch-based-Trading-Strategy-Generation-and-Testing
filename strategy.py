#!/usr/bin/env python3
"""
1D Weekly Trend Following with Volume Confirmation
Long when price closes above weekly EMA34 with volume above average, short when below.
Uses weekly EMA34 as trend filter and daily close for entry timing.
Designed for low trade frequency with strong trend-following edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA34 on weekly close
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to daily timeframe (wait for weekly bar to close)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = volume[i] > vol_ma[i]  # volume above average
        
        if position == 0:
            # Long: price closes above weekly EMA34 with volume confirmation
            if price > ema_34_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price closes below weekly EMA34 with volume confirmation
            elif price < ema_34_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price closes below weekly EMA34
            if price < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price closes above weekly EMA34
            if price > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1D_WeeklyEMA34_Trend_VolumeFilter"
timeframe = "1d"
leverage = 1.0