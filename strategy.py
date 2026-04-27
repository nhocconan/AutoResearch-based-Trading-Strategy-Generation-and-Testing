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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    dt = pd.to_datetime(open_time)
    hours = dt.hour
    session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    ema_period = 50
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= ema_period:
        ema_4h[ema_period - 1] = np.mean(close_4h[:ema_period])
        for i in range(ema_period, len(close_4h)):
            ema_4h[i] = (close_4h[i] * (2 / (ema_period + 1)) + 
                         ema_4h[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align 4h EMA to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get daily data for volatility filter (ATR)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.abs(high_1d[1:] - close_1d[:-1]),
                    np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 14
    atr_1d = np.full(len(close_1d), np.nan)
    if len(tr) >= atr_period:
        atr_1d[atr_period - 1] = np.nanmean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            atr_1d[i] = (tr[i] + (atr_period - 1) * atr_1d[i-1]) / atr_period
    
    # Align daily ATR to 1h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need EMA, ATR, volume MA
    start_idx = max(ema_period, vol_period, 100)
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not session[i] or np.isnan(ema_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        atr = atr_1d_aligned[i]
        
        if position == 0:
            # Long: Price above 4h EMA50 + volume spike + low volatility (mean reversion setup)
            if (price > ema_4h_aligned[i] and 
                vol_ratio > 1.5 and 
                atr < np.nanpercentile(atr_1d_aligned[:i+1], 30)):  # Low volatility regime
                signals[i] = size
                position = 1
            # Short: Price below 4h EMA50 + volume spike + low volatility
            elif (price < ema_4h_aligned[i] and 
                  vol_ratio > 1.5 and 
                  atr < np.nanpercentile(atr_1d_aligned[:i+1], 30)):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below 4h EMA50 OR volatility expands
            if (price < ema_4h_aligned[i] or 
                atr > np.nanpercentile(atr_1d_aligned[max(0, i-20):i+1], 80)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above 4h EMA50 OR volatility expands
            if (price > ema_4h_aligned[i] or 
                atr > np.nanpercentile(atr_1d_aligned[max(0, i-20):i+1], 80)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_EMA50_Volume_Volatility_Filter"
timeframe = "1h"
leverage = 1.0