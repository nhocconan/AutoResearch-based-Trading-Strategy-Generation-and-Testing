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
    
    # Get 1d data for ATR and Donchian
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
    
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-14:i])
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily Donchian(20)
    donch_high_20 = np.full(len(df_1d), np.nan)
    donch_low_20 = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        donch_high_20[i] = np.max(high_1d[i-20:i])
        donch_low_20[i] = np.min(low_1d[i-20:i])
    
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h_21 = np.full(len(df_12h), np.nan)
    alpha = 2 / (21 + 1)
    for i in range(len(close_12h)):
        if i == 0:
            ema_12h_21[i] = close_12h[i]
        else:
            ema_12h_21[i] = close_12h[i] * alpha + ema_12h_21[i-1] * (1 - alpha)
    
    ema_12h_21_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_21)
    
    # Calculate volume ratio: current volume / 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    volume_ratio = np.divide(volume, vol_ma_20, out=np.full(n, np.nan), where=vol_ma_20>0)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(20, 21)
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or
            np.isnan(donch_low_20_aligned[i]) or
            np.isnan(ema_12h_21_aligned[i]) or
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_14_1d_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_filter = volume_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + 12h EMA up
            if (price > donch_high_20_aligned[i] and 
                vol_filter and 
                ema_12h_21_aligned[i] > ema_12h_21_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume + 12h EMA down
            elif (price < donch_low_20_aligned[i] and 
                  vol_filter and 
                  ema_12h_21_aligned[i] < ema_12h_21_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or 12h EMA turns down
            if (price < donch_low_20_aligned[i] or 
                ema_12h_21_aligned[i] < ema_12h_21_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or 12h EMA turns up
            if (price > donch_high_20_aligned[i] or 
                ema_12h_21_aligned[i] > ema_12h_21_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_EMA21_12hTrend_v1"
timeframe = "4h"
leverage = 1.0