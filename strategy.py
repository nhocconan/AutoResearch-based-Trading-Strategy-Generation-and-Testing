#!/usr/bin/env python3
"""
4h_12h_MomentumBreakout_VolumeRegime
Hypothesis: Combining 4h momentum (price > SMA20) with 12h trend filter (EMA50) and volume confirmation (>1.5x average) creates robust signals in both bull and bear markets. The 12h EMA50 ensures we trade with the higher timeframe trend, reducing false signals during sideways periods. Volume confirmation ensures momentum is backed by participation. Designed for low trade frequency (target: 25-50/year) to minimize fee drag in 4h timeframe. Uses discrete position sizing (0.25) to reduce churn.
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
    
    # 4h SMA20 for momentum filter
    sma20 = np.zeros_like(close)
    for i in range(n):
        if i < 20:
            sma20[i] = np.mean(close[:i+1])
        else:
            sma20[i] = np.mean(close[i-20+1:i+1])
    
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
    tr[0] = tr1[0]  # First period
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
        sma20_val = sma20[i]
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
            # Long: price above SMA20 (momentum) with volume and 12h uptrend (price > 12h EMA50)
            if price > sma20_val and vol_ok and price > ema50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price below SMA20 (momentum) with volume and 12h downtrend (price < 12h EMA50)
            elif price < sma20_val and vol_ok and price < ema50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price falls below SMA20 (lost momentum) or breaks below 12h EMA50 (trend change)
            if price < sma20_val or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above SMA20 (lost momentum) or breaks above 12h EMA50 (trend change)
            if price > sma20_val or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_MomentumBreakout_VolumeRegime"
timeframe = "4h"
leverage = 1.0