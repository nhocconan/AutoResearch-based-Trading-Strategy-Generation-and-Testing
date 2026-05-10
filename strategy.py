#!/usr/bin/env python3
"""
6h_RSI_34_Pullback_1dTrend_Volume
Hypothesis: On 6h timeframe, buy pullbacks to RSI(34) in direction of 1d EMA(50) trend with volume confirmation.
Works in bull by buying dips in uptrend; works in bear by selling rallies in downtrend.
Target: 60-120 total trades over 4 years (15-30/year).
"""

name = "6h_RSI_34_Pullback_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(34) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(34, n):
        if i == 34:
            avg_gain[i] = np.mean(gain[1:35])
            avg_loss[i] = np.mean(loss[1:35])
        else:
            avg_gain[i] = (avg_gain[i-1] * 33 + gain[i]) / 34
            avg_loss[i] = (avg_loss[i-1] * 33 + loss[i]) / 34
    
    rsi = np.full(n, 50.0)
    for i in range(34, n):
        if avg_loss[i] != 0:
            rsi[i] = 100 - (100 / (1 + avg_gain[i] / avg_loss[i]))
    
    # Volume spike: current volume > 1.8x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 50, 20)  # RSI + EMA + volume warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.8 * vol_sma[i]
        
        if position == 0:
            # Long: RSI < 40 (pullback) and above 1d EMA50
            if rsi[i] < 40 and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 60 (pullback) and below 1d EMA50
            elif rsi[i] > 60 and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI > 60 (overbought) or price below 1d EMA50
            if rsi[i] > 60 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI < 40 (oversold) or price above 1d EMA50
            if rsi[i] < 40 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals