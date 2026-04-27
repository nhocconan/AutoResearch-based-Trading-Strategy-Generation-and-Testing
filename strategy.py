#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly volatility filter and volume confirmation.
# Uses weekly ATR ratio to filter low volatility regimes and avoid false breakouts.
# Works in both bull and bear markets by trading breakouts in direction of weekly trend.
# Target: 15-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for volatility filter and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ATR(20) - using high-low range
    atr_20_1w = np.full(len(df_1w), np.nan)
    for i in range(20, len(df_1w)):
        tr = np.maximum(
            high_1w[i] - low_1w[i],
            np.abs(high_1w[i] - close_1w[i-1]) if i > 0 else high_1w[i] - low_1w[i],
            np.abs(low_1w[i] - close_1w[i-1]) if i > 0 else high_1w[i] - low_1w[i]
        )
        if i == 20:
            atr_20_1w[i] = np.mean(tr)  # simple average for first value
        else:
            atr_20_1w[i] = (atr_20_1w[i-1] * 19 + tr) / 20
    
    # Weekly ATR(100) for longer-term volatility
    atr_100_1w = np.full(len(df_1w), np.nan)
    for i in range(100, len(df_1w)):
        tr = np.maximum(
            high_1w[i] - low_1w[i],
            np.abs(high_1w[i] - close_1w[i-1]) if i > 0 else high_1w[i] - low_1w[i],
            np.abs(low_1w[i] - close_1w[i-1]) if i > 0 else high_1w[i] - low_1w[i]
        )
        if i == 100:
            atr_100_1w[i] = np.mean(tr)
        else:
            atr_100_1w[i] = (atr_100_1w[i-1] * 99 + tr) / 100
    
    # Volatility ratio: current ATR / long-term ATR
    vol_ratio = np.full(len(df_1w), np.nan)
    valid = (~np.isnan(atr_20_1w)) & (~np.isnan(atr_100_1w)) & (atr_100_1w > 0)
    vol_ratio[valid] = atr_20_1w[valid] / atr_100_1w[valid]
    
    # Weekly trend: EMA(50) slope
    ema_50_1w = np.full(len(df_1w), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_1w)):
        if i < 49:
            ema_50_1w[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_50_1w[i-1]):
                ema_50_1w[i] = np.mean(close_1w[i-49:i+1])
            else:
                ema_50_1w[i] = close_1w[i] * alpha + ema_50_1w[i-1] * (1 - alpha)
    
    # Align weekly indicators to 6h
    atr_20_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_20_1w)
    atr_100_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_100_1w)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Donchian(20) channels
    upper_20 = np.full(len(df_1d), np.nan)
    lower_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        upper_20[i] = np.max(high_1d[i-19:i+1])
        lower_20[i] = np.min(low_1d[i-19:i+1])
    
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Volume spike: current volume > 2 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = max(20, 50, 100)
    
    for i in range(start_idx, n):
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or
            np.isnan(vol_ratio_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend direction from EMA slope
        if i > 0 and not np.isnan(ema_50_1w_aligned[i-1]):
            weekly_trend_up = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_trend_down = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            weekly_trend_up = False
            weekly_trend_down = False
        
        # Volatility filter: only trade in normal to high volatility (avoid low vol false breakouts)
        vol_filter = vol_ratio_aligned[i] > 0.8
        
        if position == 0:
            # Long entry: break above upper Donchian + weekly uptrend + vol filter + volume spike
            if (close[i] > upper_20_aligned[i] and 
                weekly_trend_up and 
                vol_filter and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower Donchian + weekly downtrend + vol filter + volume spike
            elif (close[i] < lower_20_aligned[i] and 
                  weekly_trend_down and 
                  vol_filter and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: break below lower Donchian or weekly trend turns down
            if (close[i] < lower_20_aligned[i] or 
                not weekly_trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above upper Donchian or weekly trend turns up
            if (close[i] > upper_20_aligned[i] or 
                not weekly_trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyVolFilter_Volume_v1"
timeframe = "6h"
leverage = 1.0