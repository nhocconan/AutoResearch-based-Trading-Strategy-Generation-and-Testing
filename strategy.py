#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d ATR for volatility regime ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr = np.zeros(len(high_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                    abs(high_1d[i] - close_1d[i-1]), 
                    abs(low_1d[i] - close_1d[i-1]))
    
    # ATR(14)
    atr = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # ATR(50)
    atr_long = np.full_like(tr, np.nan)
    if len(tr) >= 50:
        atr_long[49] = np.mean(tr[:50])
        for i in range(50, len(tr)):
            atr_long[i] = (atr_long[i-1] * 49 + tr[i]) / 50
    
    # ATR Ratio (short/long) - measures volatility regime
    atr_ratio = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if atr_long[i] > 0:
            atr_ratio[i] = atr[i] / atr_long[i]
    
    # === 1d EMA(50) for trend filter ===
    ema_50 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50[i] = alpha * close_1d[i] + (1 - alpha) * ema_50[i-1]
    else:
        for i in range(len(close_1d)):
            ema_50[i] = np.mean(close_1d[:i+1]) if i >= 0 else close_1d[0]
    
    # === Align indicators to 6h timeframe ===
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === 6h Donchian(20) breakout ===
    # Highest high of last 20 periods
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(len(high)):
        if i >= 19:
            highest_high[i] = np.max(high[i-19:i+1])
            lowest_low[i] = np.min(low[i-19:i+1])
        elif i > 0:
            highest_high[i] = np.max(high[0:i+1])
            lowest_low[i] = np.min(low[0:i+1])
        else:
            highest_high[i] = high[0]
            lowest_low[i] = low[0]
    
    # === 6h Volume confirmation ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    # Volume spike: current volume > 2x 20-period average
    vol_spike = volume > vol_ma_20 * 2.0
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Low volatility regime (ATR ratio < 0.8) + volume spike + breakout
            vol_condition = atr_ratio_aligned[i] < 0.8
            vol_spike_condition = vol_spike[i]
            
            # Long: price breaks above Donchian high AND above EMA50
            if (vol_condition and vol_spike_condition and 
                close[i] > highest_high[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low AND below EMA50
            elif (vol_condition and vol_spike_condition and 
                  close[i] < lowest_low[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit: price retests Donchian low OR volatility expands (ATR ratio > 1.2)
            if close[i] < lowest_low[i] or atr_ratio_aligned[i] > 1.2:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price retests Donchian high OR volatility expands (ATR ratio > 1.2)
            if close[i] > highest_high[i] or atr_ratio_aligned[i] > 1.2:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ATR_Volatility_Breakout_EMA50"
timeframe = "6h"
leverage = 1.0