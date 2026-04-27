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
    
    # Get daily data for 48-period EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 48:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate daily EMA(48) for trend
    ema_1d_48 = np.full(len(df_1d), np.nan)
    alpha_d = 2 / (48 + 1)
    for i in range(len(close_1d)):
        if i < 47:
            ema_1d_48[i] = np.mean(close_1d[:i+1]) if i > 0 else close_1d[i]
        else:
            if np.isnan(ema_1d_48[i-1]):
                ema_1d_48[i] = np.mean(close_1d[i-47:i+1])
            else:
                ema_1d_48[i] = close_1d[i] * alpha_d + ema_1d_48[i-1] * (1 - alpha_d)
    
    # Align daily EMA48 to 1h timeframe
    ema_1d_48_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_48)
    
    # Get 4h data for ATR calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[high_4h[0] - low_4h[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_4h = np.full(len(df_4h), np.nan)
    for i in range(len(tr_4h)):
        if i < 13:
            atr_4h[i] = np.mean(tr_4h[:i+1]) if i > 0 else tr_4h[i]
        else:
            atr_4h[i] = (atr_4h[i-1] * 13 + tr_4h[i]) / 14
    
    # Align 4h ATR to 1h timeframe
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Calculate 1h ATR(14) for volatility filter
    tr1_1h = high[1:] - low[1:]
    tr2_1h = np.abs(high[1:] - close[:-1])
    tr3_1h = np.abs(low[1:] - close[:-1])
    tr_1h = np.concatenate([[high[0] - low[0]], np.maximum(tr1_1h, np.maximum(tr2_1h, tr3_1h))])
    
    atr_1h = np.zeros(n)
    for i in range(n):
        if i < 13:
            atr_1h[i] = np.mean(tr_1h[:i+1]) if i > 0 else tr_1h[i]
        else:
            atr_1h[i] = (atr_1h[i-1] * 13 + tr_1h[i]) / 14
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 14, 48)  # volume MA needs 20, ATR needs 14, EMA needs 48
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_48_aligned[i]) or
            np.isnan(atr_4h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 1.8x average volume
        volume_confirmation = vol_ratio > 1.8
        
        # Volatility filter: only trade when 1h ATR is between 0.4x and 2.5x of 4h ATR
        vol_filter = (atr_1h[i] > atr_4h_aligned[i] * 0.4) and (atr_1h[i] < atr_4h_aligned[i] * 2.5)
        
        if position == 0:
            # Long: price above daily EMA48 with volume and volatility
            if volume_confirmation and vol_filter and price > ema_1d_48_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price below daily EMA48 with volume and volatility
            elif volume_confirmation and vol_filter and price < ema_1d_48_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below daily EMA48 or volatility too low/high
            if (price < ema_1d_48_aligned[i] or 
                atr_1h[i] < atr_4h_aligned[i] * 0.3 or
                atr_1h[i] > atr_4h_aligned[i] * 3.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # Maintain position
        elif position == -1:
            # Short exit: price crosses above daily EMA48 or volatility too low/high
            if (price > ema_1d_48_aligned[i] or 
                atr_1h[i] < atr_4h_aligned[i] * 0.3 or
                atr_1h[i] > atr_4h_aligned[i] * 3.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # Maintain position
    
    return signals

name = "1h_daily_EMA48_4hATR_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0