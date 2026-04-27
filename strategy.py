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
    
    # Get 12h data for calculations (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour Exponential Moving Average (20-period) for trend
    close_12h = df_12h['close'].values
    ema_20_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 20:
        multiplier = 2 / (20 + 1)
        ema_20_12h[19] = np.mean(close_12h[:20])
        for i in range(20, len(close_12h)):
            ema_20_12h[i] = (close_12h[i] * multiplier) + (ema_20_12h[i-1] * (1 - multiplier))
    
    # Calculate 12-hour Average True Range (14-period) for volatility
    tr_12h = np.zeros(len(close_12h))
    tr_12h[0] = high[0] - low[0] if len(high) > 0 else 0
    for i in range(1, len(close_12h)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close_12h[i-1])
        tr3 = abs(low[i] - close_12h[i-1])
        tr_12h[i] = max(tr1, tr2, tr3)
    
    atr_14_12h = np.full(len(tr_12h), np.nan)
    if len(tr_12h) >= 14:
        atr_14_12h[13] = np.mean(tr_12h[:14])
        for i in range(14, len(tr_12h)):
            atr_14_12h[i] = (tr_12h[i] * (13/14)) + (atr_14_12h[i-1] * (1/14))
    
    # Align 12h indicators to 6h timeframe
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Calculate 6-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 6
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(20, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(atr_14_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price above EMA20 and volatility expansion (ATR > 1.5x mean)
            if price > ema_20_12h_aligned[i] and atr_14_12h_aligned[i] > (np.nanmean(atr_14_12h_aligned[max(0,i-50):i]) * 1.5) and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price below EMA20 and volatility expansion (ATR > 1.5x mean)
            elif price < ema_20_12h_aligned[i] and atr_14_12h_aligned[i] > (np.nanmean(atr_14_12h_aligned[max(0,i-50):i]) * 1.5) and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below EMA20 or volatility contraction
            if price < ema_20_12h_aligned[i] or (atr_14_12h_aligned[i] < (np.nanmean(atr_14_12h_aligned[max(0,i-50):i]) * 0.5)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above EMA20 or volatility contraction
            if price > ema_20_12h_aligned[i] or (atr_14_12h_aligned[i] < (np.nanmean(atr_14_12h_aligned[max(0,i-50):i]) * 0.5)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_EMA20_ATR14_Volume"
timeframe = "6h"
leverage = 1.0