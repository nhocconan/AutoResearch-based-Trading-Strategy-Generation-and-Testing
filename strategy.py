# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h ADX trend strength + 12h EMA crossover with volume confirmation.
- ADX > 25 indicates strong trend (works in both bull/bear markets)
- 12h EMA crossover (8/21) provides entry timing with trend filter
- Volume spike confirms institutional participation
- Target: 20-40 trades/year to avoid fee drag
- Uses discrete position sizing (0.25) to minimize churn
"""

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
    
    # Get 12h data for EMA crossover (trend and timing)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA(8) and EMA(21) on 12h close
    ema8_12h = np.full(len(close_12h), np.nan)
    ema21_12h = np.full(len(close_12h), np.nan)
    
    # EMA(8)
    alpha8 = 2 / (8 + 1)
    for i in range(len(close_12h)):
        if i < 7:
            ema8_12h[i] = np.mean(close_12h[:i+1]) if i > 0 else close_12h[i]
        else:
            if np.isnan(ema8_12h[i-1]):
                ema8_12h[i] = np.mean(close_12h[i-7:i+1])
            else:
                ema8_12h[i] = close_12h[i] * alpha8 + ema8_12h[i-1] * (1 - alpha8)
    
    # EMA(21)
    alpha21 = 2 / (21 + 1)
    for i in range(len(close_12h)):
        if i < 20:
            ema21_12h[i] = np.mean(close_12h[:i+1]) if i > 0 else close_12h[i]
        else:
            if np.isnan(ema21_12h[i-1]):
                ema21_12h[i] = np.mean(close_12h[i-20:i+1])
            else:
                ema21_12h[i] = close_12h[i] * alpha21 + ema21_12h[i-1] * (1 - alpha21)
    
    ema8_12h_aligned = align_htf_to_ltf(prices, df_12h, ema8_12h)
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # Get daily data for ADX (trend strength filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily data
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nanmean(data[i-period+1:i+1])
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d > 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d > 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smooth(dx, 14)
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(30, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema8_12h_aligned[i]) or 
            np.isnan(ema21_12h_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine EMA crossover from 12h data
        # Use previous bar's values to avoid look-ahead
        if i > 0 and not np.isnan(ema8_12h_aligned[i-1]) and not np.isnan(ema21_12h_aligned[i-1]):
            ema8_prev = ema8_12h_aligned[i-1]
            ema21_prev = ema21_12h_aligned[i-1]
            ema8_curr = ema8_12h_aligned[i]
            ema21_curr = ema21_12h_aligned[i]
            
            bullish_cross = (ema8_prev <= ema21_prev) and (ema8_curr > ema21_curr)
            bearish_cross = (ema8_prev >= ema21_prev) and (ema8_curr < ema21_curr)
        else:
            bullish_cross = False
            bearish_cross = False
        
        if position == 0:
            # Long entry: bullish EMA crossover + strong trend (ADX>25) + volume spike
            if (bullish_cross and 
                adx_1d_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish EMA crossover + strong trend (ADX>25) + volume spike
            elif (bearish_cross and 
                  adx_1d_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: bearish EMA crossover or trend weakens (ADX<20)
            if (bearish_cross or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish EMA crossover or trend weakens (ADX<20)
            if (bullish_cross or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_EMACross_12h_Volume_v1"
timeframe = "6h"
leverage = 1.0