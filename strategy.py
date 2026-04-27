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
    
    # Get daily data for pivot calculation (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate classic pivot points: P = (H+L+C)/3, R1 = 2*P-L, S1 = 2*P-H
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get weekly data for trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly EMA(20)
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
    
    # Align weekly EMA20 to 4h timeframe
    ema_1w_20_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_20)
    
    # Get 4h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.zeros(n)
    for i in range(n):
        if i < 13:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[i]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 14)  # volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_1w_20_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        # Trend filter: price above/below weekly EMA20
        uptrend = price > ema_1w_20_aligned[i]
        downtrend = price < ema_1w_20_aligned[i]
        
        if position == 0:
            # Long: price crosses above S1 with volume and uptrend
            if volume_confirmation and uptrend and price > s1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1 with volume and downtrend
            elif volume_confirmation and downtrend and price < r1_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below pivot or trend reverses
            if price < pivot_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price crosses above pivot or trend reverses
            if price > pivot_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_1D_Pivot_S1R1_Breakout_1wEMA20_Trend_Volume"
timeframe = "4h"
leverage = 1.0