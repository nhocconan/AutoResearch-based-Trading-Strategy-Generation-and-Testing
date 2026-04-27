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
    
    # Get daily data for ATR and close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(len(tr_1d)):
        if i < 13:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Get weekly close for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_21 = np.full(len(close_1w), np.nan)
    alpha_w = 2 / (21 + 1)
    for i in range(len(close_1w)):
        if i < 20:
            ema_1w_21[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_1w_21[i-1]):
                ema_1w_21[i] = np.mean(close_1w[i-20:i+1])
            else:
                ema_1w_21[i] = close_1w[i] * alpha_w + ema_1w_21[i-1] * (1 - alpha_w)
    
    # Align indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_1w_21_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_21)
    
    # Calculate 4-hour ATR for stop loss
    tr1_4h = high[1:] - low[1:]
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[high[0] - low[0]], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    
    atr_4h = np.full(n, np.nan)
    for i in range(n):
        if i < 13:
            atr_4h[i] = np.mean(tr_4h[:i+1]) if i > 0 else tr_4h[i]
        else:
            atr_4h[i] = (atr_4h[i-1] * 13 + tr_4h[i]) / 14
    
    # Calculate 4-hour volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 21)  # volume MA needs 20, weekly EMA needs 21
    
    for i in range(start_idx, n):
        if (np.isnan(atr_1d_aligned[i]) or
            np.isnan(ema_1w_21_aligned[i]) or
            np.isnan(atr_4h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 2.0x average volume (strict to reduce trades)
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long: price > EMA21 weekly + volatility expansion + volume spike
            if (volume_confirmation and 
                price > ema_1w_21_aligned[i] and 
                atr_1d_aligned[i] > atr_1d_aligned[i-1] * 1.2):  # volatility expansion
                signals[i] = 0.25
                position = 1
            # Short: price < EMA21 weekly + volatility expansion + volume spike
            elif (volume_confirmation and 
                  price < ema_1w_21_aligned[i] and 
                  atr_1d_aligned[i] > atr_1d_aligned[i-1] * 1.2):  # volatility expansion
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price < EMA21 weekly or volatility contraction
            if (price < ema_1w_21_aligned[i] or 
                atr_1d_aligned[i] < atr_1d_aligned[i-1] * 0.8):  # volatility contraction
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price > EMA21 weekly or volatility contraction
            if (price > ema_1w_21_aligned[i] or 
                atr_1d_aligned[i] < atr_1d_aligned[i-1] * 0.8):  # volatility contraction
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_VolatilityExpansion_EMA21Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0