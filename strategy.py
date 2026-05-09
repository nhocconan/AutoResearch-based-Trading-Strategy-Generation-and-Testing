#!/usr/bin/env python3
# 6h_VIX_Filtered_Momentum
# Hypothesis: Use VIX-like volatility index (realized volatility ratio) to filter momentum trades.
# In high volatility regimes, momentum fails; in low volatility, momentum persists.
# Long when: price > 6h EMA(20) AND volatility ratio < 0.8 (low vol)
# Short when: price < 6h EMA(20) AND volatility ratio < 0.8 (low vol)
# Exit when volatility ratio > 1.2 (high vol) or opposite EMA cross
# Uses 1d ATR ratio (current ATR/20-period MA of ATR) as volatility filter
# Designed for 6h to avoid whipsaws in volatile markets while capturing trends in calm periods

name = "6h_VIX_Filtered_Momentum"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate 1d ATR(14) and its 20-period MA for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need 14 for ATR + 20 for MA
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr[13] = np.nanmean(tr[0:14])
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # ATR 20-period MA
    atr_ma = np.full_like(atr, np.nan)
    if len(atr) >= 34:  # 14 + 20
        atr_ma[33] = np.nanmean(atr[14:34])
        for i in range(34, len(atr)):
            atr_ma[i] = (atr_ma[i-1] * 19 + atr[i]) / 20
    
    # Volatility ratio: current ATR / MA of ATR
    vol_ratio = np.full_like(atr, np.nan)
    valid = (~np.isnan(atr)) & (~np.isnan(atr_ma)) & (atr_ma != 0)
    vol_ratio[valid] = atr[valid] / atr_ma[valid]
    
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # Calculate 6h EMA(20)
    ema_20 = np.full_like(close, np.nan)
    if len(close) >= 20:
        ema_20[19] = np.nanmean(clone := close[0:20])
        for i in range(20, len(close)):
            ema_20[i] = (ema_20[i-1] * 19 + close[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure ATR MA is ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above EMA20 AND low volatility (vol ratio < 0.8)
            if close[i] > ema_20[i] and vol_ratio_aligned[i] < 0.8:
                signals[i] = 0.25
                position = 1
            # Enter short: price below EMA20 AND low volatility (vol ratio < 0.8)
            elif close[i] < ema_20[i] and vol_ratio_aligned[i] < 0.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below EMA20 OR high volatility (vol ratio > 1.2)
            if close[i] < ema_20[i] or vol_ratio_aligned[i] > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above EMA20 OR high volatility (vol ratio > 1.2)
            if close[i] > ema_20[i] or vol_ratio_aligned[i] > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals