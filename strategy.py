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
    
    # Get 1d data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day ATR (14-period) for volatility and stop loss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.zeros(len(close_1d))
    atr_1d = np.zeros(len(close_1d))
    if len(close_1d) >= 2:
        tr1[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr1[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        # Calculate ATR using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        if len(tr1) >= 14:
            atr_1d[13] = np.mean(tr1[:14])
            for i in range(14, len(tr1)):
                atr_1d[i] = (atr_1d[i-1] * 13 + tr1[i]) / 14
    
    # Calculate 1-day Exponential Moving Average (34-period) for trend
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        multiplier = 2 / (34 + 1)
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * multiplier) + (ema_34_1d[i-1] * (1 - multiplier))
    
    # Align 1d indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 4
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(34, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price above EMA34 and breaks with volume
            if price > ema_34_1d_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price below EMA34 and breaks with volume
            elif price < ema_34_1d_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below EMA34 or volatility spike (potential reversal)
            if price < ema_34_1d_aligned[i] or (vol_ratio > 2.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above EMA34 or volatility spike (potential reversal)
            if price > ema_34_1d_aligned[i] or (vol_ratio > 2.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_EMA34_ATR_Volume"
timeframe = "4h"
leverage = 1.0