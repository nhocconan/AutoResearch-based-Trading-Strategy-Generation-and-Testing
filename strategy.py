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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA21
    ema_period = 21
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                        ema_1w[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align weekly EMA to daily
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no previous close
    
    # ATR using Wilder's smoothing
    atr_period = 14
    atr_1d = np.full(len(tr), np.nan)
    if len(tr) >= atr_period:
        atr_1d[atr_period - 1] = np.nanmean(tr[1:atr_period])  # Skip first NaN
        for i in range(atr_period, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Align daily ATR to daily (no change, but for consistency)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily volume filter: current volume > 1.8x 20-day average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly EMA, daily ATR, and volume MA
    start_idx = max(21, 14, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        atr = atr_1d_aligned[i]
        
        if position == 0:
            # Long: Price above weekly EMA21 with volume confirmation and low volatility
            if (price > ema_1w_aligned[i] and 
                vol_ratio > 1.8 and
                atr < np.nanmedian(atr_1d_aligned[max(0, i-30):i]) * 1.5):  # Volatility filter
                signals[i] = size
                position = 1
            # Short: Price below weekly EMA21 with volume confirmation and low volatility
            elif (price < ema_1w_aligned[i] and 
                  vol_ratio > 1.8 and
                  atr < np.nanmedian(atr_1d_aligned[max(0, i-30):i]) * 1.5):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below weekly EMA21 or volatility spike
            if (price < ema_1w_aligned[i] or 
                atr > np.nanmedian(atr_1d_aligned[max(0, i-30):i]) * 2.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above weekly EMA21 or volatility spike
            if (price > ema_1w_aligned[i] or 
                atr > np.nanmedian(atr_1d_aligned[max(0, i-30):i]) * 2.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyEMA21_VolumeVolatility_Filter"
timeframe = "1d"
leverage = 1.0