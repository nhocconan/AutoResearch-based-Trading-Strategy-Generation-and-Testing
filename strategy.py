#!/usr/bin/env python3
"""
4h_12h_Pullback_TrendFollow
Hypothesis: In trending markets, price pulls back to the 12-period EMA on the 4h chart before resuming trend. Using 12h EMA50 as trend filter ensures we only trade in the direction of higher timeframe momentum. Volume confirmation (>1.5x average) filters weak moves. Designed for low trade frequency (~20-40/year) to minimize fee drag. Works in bull markets (buy pullbacks in uptrends) and bear markets (sell rallies in downtrends).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data once for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = np.zeros_like(close_12h)
    ema50_12h[0] = close_12h[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_12h)):
        ema50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema50_12h[i-1]
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA12 for pullback entries
    ema12 = np.zeros_like(close)
    ema12[0] = close[0]
    alpha12 = 2.0 / (12 + 1)
    for i in range(1, len(close)):
        ema12[i] = alpha12 * close[i] + (1 - alpha12) * ema12[i-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.zeros_like(close)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-14:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if NaN in critical values
        if np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50 = ema50_12h_aligned[i]
        ema12_val = ema12[i]
        vol_ok = volume_filter[i]
        atr_val = atr[i]
        
        # Stoploss: 2.5 * ATR from entry
        if position == 1 and price < entry_price - 2.5 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.5 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price pulls back to EMA12 in uptrend (price > 12h EMA50) with volume
            if price >= ema12_val and price > ema50 and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price rallies to EMA12 in downtrend (price < 12h EMA50) with volume
            elif price <= ema12_val and price < ema50 and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price breaks below EMA12 (end of pullback/resume) or trend change
            if price < ema12_val or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above EMA12 (end of rally/resume) or trend change
            if price > ema12_val or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Pullback_TrendFollow"
timeframe = "4h"
leverage = 1.0