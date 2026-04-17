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
    
    # === 1d EMA (34-period) for trend direction ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA with proper Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / 34
    ema_34 = np.full_like(close_1d, np.nan)
    ema_34[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema_34[i] = alpha * close_1d[i] + (1 - alpha) * ema_34[i-1]
    
    # === 1d ATR (14-period) for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing for ATR
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Align all indicators to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        vol_ma_20 = np.zeros_like(volume)
        for j in range(len(volume)):
            if j >= 19:
                vol_ma_20[j] = np.mean(volume[j-19:j+1])
            else:
                vol_ma_20[j] = np.mean(volume[max(0, j-9):j+1]) if j > 0 else volume[0]
        vol_confirm = volume[i] > vol_ma_20[i] * 1.3
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above EMA34 + volatility filter (ATR > 0.5% price) + volume
            if (close[i] > ema_34_aligned[i] and 
                atr_14_aligned[i] > 0.005 * close[i] and  # volatility filter
                vol_confirm):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below EMA34 + volatility filter + volume
            elif (close[i] < ema_34_aligned[i] and 
                  atr_14_aligned[i] > 0.005 * close[i] and  # volatility filter
                  vol_confirm):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below EMA34
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA34
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "EMA34_ATR_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0