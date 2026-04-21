#!/usr/bin/env python3
"""
12h_Donchian20_VolumeRegime
Hypothesis: 12h Donchian(20) breakouts with volume confirmation and 1w trend filter capture strong moves in both bull and bear markets. 
1w EMA50 ensures we trade with the higher timeframe trend, reducing false signals during sideways periods. 
Volume > 1.5x average confirms breakout strength. Designed for low trade frequency (target: 15-30/year) to minimize fee drift in 12h timeframe.
Uses discrete position sizing (0.25) to reduce churn. Works in bull (breakouts continue) and bear (breakdowns continue) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1w data once for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = np.zeros_like(close_1w)
    ema50_1w[0] = close_1w[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1w)):
        ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    
    # Align 1w EMA50 to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian(20) channels
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(n):
        if i >= 20:
            upper[i] = np.max(high[i-20:i])
            lower[i] = np.min(low[i-20:i])
        else:
            upper[i] = np.max(high[:i+1]) if i > 0 else high[i]
            lower[i] = np.min(low[:i+1]) if i > 0 else low[i]
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.full(n, np.nan)
    for i in range(n):
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
    tr[0] = tr1[0]  # First period
    atr = np.full(n, np.nan)
    for i in range(n):
        if i >= 14:
            atr[i] = np.mean(tr[i-14:i])
        else:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50 = ema50_1w_aligned[i]
        up = upper[i]
        low_ch = lower[i]
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
            # Long: price breaks above Donchian upper with volume and 1w uptrend (price > 1w EMA50)
            if price > up and vol_ok and price > ema50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower with volume and 1w downtrend (price < 1w EMA50)
            elif price < low_ch and vol_ok and price < ema50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price falls below Donchian lower or breaks below 1w EMA50 (trend change)
            if price < low_ch or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above Donchian upper or breaks above 1w EMA50 (trend change)
            if price > up or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeRegime"
timeframe = "12h"
leverage = 1.0