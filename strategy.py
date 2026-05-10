#!/usr/bin/env python3
"""
4h_RSI_Range_Filter_Momentum
Hypothesis: Enter long when RSI(14) > 55 and price > 4h EMA(20); enter short when RSI(14) < 45 and price < 4h EMA(20). Use 1d ADX(14) > 20 to confirm trending regime and avoid ranging markets. Exit on opposite RSI crossover (RSI < 50 for long exit, RSI > 50 for short). This captures momentum in trending markets while avoiding false signals in ranges. RSI provides objective entry/exit levels, EMA(20) filters for trend alignment, and ADX ensures sufficient trend strength. Target: 20-40 trades/year.
"""

name = "4h_RSI_Range_Filter_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA(20)
    ema_20 = np.full_like(close, np.nan)
    if len(close) >= 20:
        ema_20[19] = np.mean(close[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close)):
            ema_20[i] = alpha * close[i] + (1 - alpha) * ema_20[i-1]
    
    # 1d ADX(14) for trend strength
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) >= period + 1:
            smoothed[period] = np.nansum(arr[1:period+1])
            for i in range(period+1, len(arr)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    atr = smooth_wilder(tr, 14)
    dm_plus_smooth = smooth_wilder(dm_plus, 14)
    dm_minus_smooth = smooth_wilder(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.divide(dm_plus_smooth, atr, out=np.full_like(dm_plus_smooth, np.nan), where=atr!=0) * 100
    di_minus = np.divide(dm_minus_smooth, atr, out=np.full_like(dm_minus_smooth, np.nan), where=atr!=0) * 100
    
    # DX and ADX
    dx = np.divide(np.abs(di_plus - di_minus), (di_plus + di_minus), 
                   out=np.full_like(di_plus, np.nan), where=(di_plus + di_minus)!=0) * 100
    adx = smooth_wilder(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema_20[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_aligned[i] > 20
        
        if position == 0:
            # Long: RSI > 55, price above EMA20, in trending market
            if rsi[i] > 55 and close[i] > ema_20[i] and trending:
                signals[i] = 0.25
                position = 1
            # Short: RSI < 45, price below EMA20, in trending market
            elif rsi[i] < 45 and close[i] < ema_20[i] and trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI < 50 (momentum fading)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI > 50 (momentum fading)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals