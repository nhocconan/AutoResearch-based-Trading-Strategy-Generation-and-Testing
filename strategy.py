#!/usr/bin/env python3
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
    
    # === 1d ATR for volatility regime ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr_1d = np.full_like(close_1d, np.nan)
    if len(tr) >= 14:
        atr_1d[13] = np.nanmean(tr[1:15])  # skip first NaN
        for i in range(14, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # === 1d EMA(50) for trend filter ===
    ema_50 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50[i] = alpha * close_1d[i] + (1 - alpha) * ema_50[i-1]
    
    # === Align indicators to 4h timeframe ===
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === 4h Donchian channel (20-period) ===
    # Highest high over last 20 periods
    highest_high = np.full_like(close, np.nan)
    lowest_low = np.full_like(close, np.nan)
    for i in range(len(close)):
        if i >= 19:
            highest_high[i] = np.max(close[i-19:i+1])
            lowest_low[i] = np.min(close[i-19:i+1])
        elif i > 0:
            highest_high[i] = np.max(close[0:i+1])
            lowest_low[i] = np.min(close[0:i+1])
        else:
            highest_high[i] = close[0]
            lowest_low[i] = close[0]
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: breakout above Donchian high + ATR filter + above EMA50
            if (close[i] > highest_high[i] and 
                atr_1d_aligned[i] > 0 and  # volatility present
                close[i] > ema_50_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: breakdown below Donchian low + ATR filter + below EMA50
            elif (close[i] < lowest_low[i] and 
                  atr_1d_aligned[i] > 0 and  # volatility present
                  close[i] < ema_50_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: breakdown below Donchian low OR below EMA50
            if close[i] < lowest_low[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: breakout above Donchian high OR above EMA50
            if close[i] > highest_high[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATR_VolumeFilter_EMA50"
timeframe = "4h"
leverage = 1.0