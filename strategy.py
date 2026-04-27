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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly EMA(20) for trend
    ema_1w_20 = np.full(len(df_1w), np.nan)
    alpha_w = 2 / (20 + 1)
    for i in range(len(close_1w)):
        if i < 19:
            ema_1w_20[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_1w_20[i-1]):
                ema_1w_20[i] = np.mean(close_1w[i-19:i+1])
            else:
                ema_1w_20[i] = close_1w[i] * alpha_w + ema_1w_20[i-1] * (1 - alpha_w)
    
    # Align weekly EMA20 to 12h timeframe
    ema_1w_20_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_20)
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(len(tr_1d)):
        if i < 13:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Align daily ATR to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12h ATR(14) for volatility filter
    tr1_12h = high[1:] - low[1:]
    tr2_12h = np.abs(high[1:] - close[:-1])
    tr3_12h = np.abs(low[1:] - close[:-1])
    tr_12h = np.concatenate([[high[0] - low[0]], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    
    atr_12h = np.zeros(n)
    for i in range(n):
        if i < 13:
            atr_12h[i] = np.mean(tr_12h[:i+1]) if i > 0 else tr_12h[i]
        else:
            atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
    
    # Calculate volume average (10-period)
    vol_ma_10 = np.full(n, np.nan)
    for i in range(10, n):
        vol_ma_10[i] = np.mean(volume[i-10:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(10, 14, 20)  # volume MA needs 10, ATR needs 14, EMA needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_20_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_10[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_10[i] if vol_ma_10[i] > 0 else 0
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        # ATR volatility filter: only trade when 12h ATR is above 50% of daily ATR
        vol_filter = atr_12h[i] > atr_1d_aligned[i] * 0.5
        
        if position == 0:
            # Long: price above weekly EMA20 with volume and volatility
            if volume_confirmation and vol_filter and price > ema_1w_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA20 with volume and volatility
            elif volume_confirmation and vol_filter and price < ema_1w_20_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below weekly EMA20 or volatility drops
            if price < ema_1w_20_aligned[i] or atr_12h[i] < atr_1d_aligned[i] * 0.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price crosses above weekly EMA20 or volatility drops
            if price > ema_1w_20_aligned[i] or atr_12h[i] < atr_1d_aligned[i] * 0.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "12h_weekly_EMA20_Trend_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0