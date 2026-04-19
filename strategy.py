#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Pivot_R1S1_Breakout_Volume_ATRFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r1_1w = close_1w + range_1w * 1.1 / 12.0
    s1_1w = close_1w - range_1w * 1.1 / 12.0
    
    # Align pivot levels to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_12h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_12h = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Get 1d data for volume context
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_12h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 12h ATR for volatility and stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_12h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Volume confirmation: current volume > 1.5x 1d EMA20
    vol_confirm = volume > 1.5 * vol_ma_12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or \
           np.isnan(atr_12h[i]) or np.isnan(vol_ma_12h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pivot = pivot_12h[i]
        r1 = r1_12h[i]
        s1 = s1_12h[i]
        atr = atr_12h[i]
        
        if position == 0:
            # Long: Price breaks above R1 + volume confirmation
            if price > r1 and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + volume confirmation
            elif price < s1 and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below pivot OR ATR stop (1.5x ATR from entry)
            if price < pivot or price < (high[i] - 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above pivot OR ATR stop (1.5x ATR from entry)
            if price > pivot or price > (low[i] + 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals