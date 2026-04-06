#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_r4s4_breakout_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for volatility
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(1, len(high_1d)):
        # Previous day's range
        prev_range = high_1d[i-1] - low_1d[i-1]
        if prev_range <= 0:
            continue
        camarilla_high[i] = close_1d[i-1] + (prev_range * 1.1 / 2)
        camarilla_low[i] = close_1d[i-1] - (prev_range * 1.1 / 2)
        camarilla_r4[i] = close_1d[i-1] + (prev_range * 1.1 * 2)
        camarilla_s4[i] = close_1d[i-1] - (prev_range * 1.1 * 2)
    
    # Align to 6h timeframe
    camarilla_high_6h = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_6h = align_htf_to_ltf(prices, df_1d, camarilla_low)
    camarilla_r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume filter: volume > 1.5x average of last 24 periods (4 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    
    start = max(30, 24)
    
    for i in range(start, n):
        if np.isnan(atr[i]) or np.isnan(camarilla_r4_6h[i]) or np.isnan(camarilla_s4_6h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        if position == 1:
            # Exit on S4 touch or stoploss
            if close[i] <= camarilla_s4_6h[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on R4 touch or stoploss
            if close[i] >= camarilla_r4_6h[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Breakout entries: long above R4, short below S4
            if close[i] > camarilla_r4_6h[i] and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif close[i] < camarilla_s4_6h[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals