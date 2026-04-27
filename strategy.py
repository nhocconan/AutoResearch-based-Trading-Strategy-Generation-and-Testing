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
    
    # Get daily data for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on daily close using pandas EWMA with min_periods
    ema_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Calculate 14-period ATR using Wilder's smoothing equivalent
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_1d[i] = np.mean(tr[1:15])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align daily indicators to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: current volume > 1.5x 24-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 24
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA, ATR, and volume MA
    start_idx = max(34, 14, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        atr = atr_1d_aligned[i]
        
        if position == 0:
            # Long: Price above daily EMA34 with volume confirmation
            if (price > ema_1d_aligned[i] and 
                vol_ratio > 1.5):
                signals[i] = size
                position = 1
            # Short: Price below daily EMA34 with volume confirmation
            elif (price < ema_1d_aligned[i] and 
                  vol_ratio > 1.5):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below daily EMA34 or ATR-based stop
            if (price < ema_1d_aligned[i] or 
                price < close[i-1] - 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above daily EMA34 or ATR-based stop
            if (price > ema_1d_aligned[i] or 
                price > close[i-1] + 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_EMA34_Volume_Trend_Filter"
timeframe = "12h"
leverage = 1.0