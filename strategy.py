#!/usr/bin/env python3
"""
1h_Donchian_Breakout_4hTrend_VolumeFilter
Hypothesis: Trade 1h Donchian breakouts (20-period) only when aligned with 4h trend (EMA50) and volume confirmation (>1.5x average). This uses higher timeframe for direction (reducing false signals) and lower timeframe for entry timing. Volume ensures breakout has participation. Designed for low trade frequency (target: 15-30/year) to minimize fee drag in 1h timeframe. Uses discrete position sizing (0.20).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data once for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = np.zeros_like(close_4h)
    ema50_4h[0] = close_4h[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_4h)):
        ema50_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema50_4h[i-1]
    
    # Align 4h EMA50 to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Main timeframe data (1h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 20:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
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
            atr[i] = np.mean(tr[:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_avg[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50 = ema50_4h_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
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
            # Long: price breaks above Donchian high with volume and 4h uptrend (price > 4h EMA50)
            if price > upper and vol_ok and price > ema50:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below Donchian low with volume and 4h downtrend (price < 4h EMA50)
            elif price < lower and vol_ok and price < ema50:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price falls below Donchian low or breaks below 4h EMA50 (trend change)
            if price < lower or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price rises above Donchian high or breaks above 4h EMA50 (trend change)
            if price > upper or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian_Breakout_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0