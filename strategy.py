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
    
    # Get 1d data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_period = 14
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(atr_period, len(tr)):
        atr_1d[i] = np.nanmean(tr[i-atr_period+1:i+1])
    
    # Get 1w data for trend filter (weekly EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50
    ema_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align indicators to daily timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volatility filter: current ATR > 1.2x 20-day average ATR
    atr_ma = np.full(n, np.nan)
    atr_ma_period = 20
    for i in range(atr_ma_period, n):
        if not np.isnan(atr_1d_aligned[i]):
            atr_ma[i] = np.nanmean(atr_1d_aligned[i-atr_ma_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR, EMA, and ATR MA
    start_idx = max(atr_period, atr_ma_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_ratio = atr_1d_aligned[i] / atr_ma[i] if atr_ma[i] > 0 else 0
        
        if position == 0:
            # Long: High volatility + price above weekly EMA50
            if (atr_ratio > 1.2 and 
                price > ema_1w_aligned[i]):
                signals[i] = size
                position = 1
            # Short: High volatility + price below weekly EMA50
            elif (atr_ratio > 1.2 and 
                  price < ema_1w_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Low volatility OR price crosses below weekly EMA50
            if (atr_ratio < 0.8 or 
                price < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Low volatility OR price crosses above weekly EMA50
            if (atr_ratio < 0.8 or 
                price > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_ATR_Volatility_Filter_WeeklyEMA50"
timeframe = "1d"
leverage = 1.0