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
    
    # Get daily data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1d = np.zeros(len(df_1d))
    for i in range(len(tr_1d)):
        if i < 13:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Align daily ATR to 6h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6-day average ATR for volatility regime
    atr_ma_6d = np.full(len(atr_1d), np.nan)
    for i in range(6, len(atr_1d)):
        atr_ma_6d[i] = np.mean(atr_1d[i-6:i])
    
    atr_ma_6d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_6d)
    
    # Calculate 6h ATR(10) for entry/exit
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.concatenate([[high[0] - low[0]], np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))])
    
    atr_6h = np.zeros(n)
    for i in range(n):
        if i < 9:
            atr_6h[i] = np.mean(tr_6h[:i+1]) if i > 0 else tr_6h[i]
        else:
            atr_6h[i] = (atr_6h[i-1] * 9 + tr_6h[i]) / 10
    
    # Calculate 6h EMA(21) for trend filter
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate volume spike detector (20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(21, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(atr_1d_aligned[i]) or
            np.isnan(atr_ma_6d_aligned[i]) or
            np.isnan(ema_21[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        atr_ratio = atr_1d_aligned[i] / atr_ma_6d_aligned[i] if atr_ma_6d_aligned[i] > 0 else 1
        
        # Volatility filter: only trade when current ATR > 1.2x 6-day average (high vol regime)
        high_volatility = atr_ratio > 1.2
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: price > EMA21 + volatility + volume
            if (high_volatility and volume_confirmation and 
                price > ema_21[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < EMA21 + volatility + volume
            elif (high_volatility and volume_confirmation and 
                  price < ema_21[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price < EMA21 or volatility drops
            if (price < ema_21[i] or atr_ratio < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > EMA21 or volatility drops
            if (price > ema_21[i] or atr_ratio < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ATR_Volatility_Volume_EMA21_Filter"
timeframe = "6h"
leverage = 1.0