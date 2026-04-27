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
    
    # Get daily data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period) for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = close_1d[0]
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d_prev)
    tr3 = np.abs(low_1d - close_1d_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr_14_1d[13] = np.mean(tr[1:15])
        for i in range(14, len(tr)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 1-day moving average (50-period) for trend filter
    ma_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        for i in range(49, len(close_1d)):
            ma_50_1d[i] = np.mean(close_1d[i-49:i+1])
    
    # Align daily indicators to 6h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    ma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ma_50_1d)
    
    # Calculate 6-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 6
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(14, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(ma_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.8x average volume
        vol_filter = vol_ratio > 1.8
        
        # Trend filter: price above/below 50-day MA
        uptrend = price > ma_50_1d_aligned[i]
        downtrend = price < ma_50_1d_aligned[i]
        
        if position == 0:
            # Long: Volatility expansion + uptrend + volume spike
            if atr_14_1d_aligned[i] > atr_14_1d_aligned[i-1] * 1.1 and uptrend and vol_filter:
                signals[i] = size
                position = 1
            # Short: Volatility expansion + downtrend + volume spike
            elif atr_14_1d_aligned[i] > atr_14_1d_aligned[i-1] * 1.1 and downtrend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Volatility contraction or trend reversal
            if atr_14_1d_aligned[i] < atr_14_1d_aligned[i-1] * 0.9 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Volatility contraction or trend reversal
            if atr_14_1d_aligned[i] < atr_14_1d_aligned[i-1] * 0.9 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Volatility_Expansion_Trend"
timeframe = "6h"
leverage = 1.0