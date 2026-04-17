#!/usr/bin/env python3
"""
12h Camarilla Pivot + Volume Spike + 1d EMA50 Trend Filter
Long when price breaks above Camarilla H3 on volume > 1.5x 20-period average and price > 1d EMA50.
Short when price breaks below Camarilla L3 on volume > 1.5x average and price < 1d EMA50.
Exit when price returns to Camarilla Pivot level or opposite breakout occurs.
Designed for 12h to capture breakouts with trend alignment, low trade frequency (~15-25/year), and avoid false breakouts in ranging markets.
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
    
    # Get 1d data for Camarilla pivot calculation and EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla pivot levels based on previous day's OHLC
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    
    # Pivot point (PP)
    pp = (prev_high + prev_low + prev_close) / 3
    # Range
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    h3 = pp + (range_hl * 1.1 / 4)  # Resistance level 3
    l3 = pp - (range_hl * 1.1 / 4)  # Support level 3
    # Also calculate pivot for exit
    pivot = pp
    
    # Align Camarilla levels to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1d, h3.values)
    l3_12h = align_htf_to_ltf(prices, df_1d, l3.values)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot.values)
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: 20-period volume MA on 12h (using current timeframe volume)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or np.isnan(pivot_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price breaks above H3 with volume spike and price > EMA50
            if price > h3_12h[i] and close[i-1] <= h3_12h[i-1] and \
               vol > 1.5 * vol_ma and price > ema_50_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 with volume spike and price < EMA50
            elif price < l3_12h[i] and close[i-1] >= l3_12h[i-1] and \
                 vol > 1.5 * vol_ma and price < ema_50_12h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot level OR breaks below L3 (contrarian)
            if price <= pivot_12h[i] or \
               (price < l3_12h[i] and close[i-1] >= l3_12h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot level OR breaks above H3 (contrarian)
            if price >= pivot_12h[i] or \
               (price > h3_12h[i] and close[i-1] <= h3_12h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Volume_EMA50"
timeframe = "12h"
leverage = 1.0