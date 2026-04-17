#!/usr/bin/env python3
"""
4h Camarilla Pivot Breakout + Volume Spike + 1d EMA Trend
Long: Price breaks above H3 with volume > 1.5x 4h volume SMA(20) and price > 1d EMA(34)
Short: Price breaks below L3 with volume > 1.5x 4h volume SMA(20) and price < 1d EMA(34)
Exit: Opposite pivot level (L3 for long, H3 for short) or trend reversal (price crosses 1d EMA)
Uses Camarilla levels from daily timeframe for institutional support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and EMA trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (H3, L3) from previous day
    # H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    rang = high_1d - low_1d
    h3 = close_1d + 1.1 * rang / 6
    l3 = close_1d - 1.1 * rang / 6
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h volume SMA(20)
    vol_sma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(20, 34)  # need volume SMA and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_sma_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_4h[i]
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        ema_val = ema_34_aligned[i]
        
        if position == 0:
            # Long: Price breaks above H3 + volume spike + above daily EMA
            if price > h3_val and vol > 1.5 * vol_sma_val and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L3 + volume spike + below daily EMA
            elif price < l3_val and vol > 1.5 * vol_sma_val and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below L3 or crosses below daily EMA
            if price < l3_val or price < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above H3 or crosses above daily EMA
            if price > h3_val or price > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0