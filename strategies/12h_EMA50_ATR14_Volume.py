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
    
    # Get 1d data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day Exponential Moving Average (50-period) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * multiplier) + (ema_50_1d[i-1] * (1 - multiplier))
    
    # Calculate 1-day Average True Range (14-period) for volatility
    if len(df_1d) >= 14:
        tr = np.zeros(len(df_1d))
        tr[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
        for i in range(1, len(df_1d)):
            tr[i] = max(
                df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
            )
        atr_14_1d = np.full(len(df_1d), np.nan)
        atr_14_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_14_1d[i] = (tr[i] * (1/14)) + (atr_14_1d[i-1] * (13/14))
    else:
        atr_14_1d = np.full(len(df_1d), np.nan)
    
    # Align 1d indicators to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 12-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 12
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(50, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price above EMA50 and ATR-based breakout with volume
            if price > ema_50_1d_aligned[i] and price > high[i-1] + 0.5 * atr_14_1d_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price below EMA50 and ATR-based breakdown with volume
            elif price < ema_50_1d_aligned[i] and price < low[i-1] - 0.5 * atr_14_1d_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below EMA50 or volatility spike (potential reversal)
            if price < ema_50_1d_aligned[i] or (vol_ratio > 2.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above EMA50 or volatility spike (potential reversal)
            if price > ema_50_1d_aligned[i] or (vol_ratio > 2.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_EMA50_ATR14_Volume"
timeframe = "12h"
leverage = 1.0