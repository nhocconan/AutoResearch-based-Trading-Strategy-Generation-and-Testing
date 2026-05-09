#!/usr/bin/env python3
# 1D_1W_Momentum_Follow
# Hypothesis: Trend following on daily timeframe using 50-day EMA for direction, with weekly trend confirmation to avoid whipsaws in ranging markets. Weekly ADX > 25 confirms strong trend. Entry when price closes above/below 50-day EMA with weekly trend alignment. Exit when price crosses back below/above EMA or weekly trend weakens. Designed for 10-25 trades/year on daily timeframe to minimize fee drag while capturing major trends in both bull and bear markets.

name = "1D_1W_Momentum_Follow"
timeframe = "1d"
leverage = 1.0

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
    
    # Get daily data (same as primary timeframe for EMA calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 50-day EMA for trend direction
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # Align 50-day EMA to daily timeframe (no alignment needed as same timeframe)
    ema_50_1d_aligned = ema_50_1d  # Already on daily timeframe
    
    # Get weekly data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:  # Need at least 14 periods for ADX
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) for weekly trend strength
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    def wildeR_smoothing(values, period):
        smoothed = np.full_like(values, np.nan)
        if len(values) >= period:
            smoothed[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                if not np.isnan(smoothed[i-1]):
                    smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    atr_1w = wildeR_smoothing(tr, 14)
    dm_plus_smoothed = wildeR_smoothing(dm_plus, 14)
    dm_minus_smoothed = wildeR_smoothing(dm_minus, 14)
    
    # Directional Indicators
    di_plus = np.full_like(close_1w, np.nan)
    di_minus = np.full_like(close_1w, np.nan)
    dx = np.full_like(close_1w, np.nan)
    
    valid = (~np.isnan(atr_1w)) & (atr_1w != 0)
    di_plus[valid] = 100 * dm_plus_smoothed[valid] / atr_1w[valid]
    di_minus[valid] = 100 * dm_minus_smoothed[valid] / atr_1w[valid]
    
    dx_valid = (~np.isnan(di_plus)) & (~np.isnan(di_minus)) & ((di_plus + di_minus) != 0)
    dx[dx_valid] = 100 * np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])
    
    # ADX is smoothed DX
    adx_1w = wildeR_smoothing(dx, 14)
    
    # Align weekly ADX to daily timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 days for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price above EMA50 AND weekly ADX > 25 (strong trend)
            if close[i] > ema_50_1d_aligned[i] and adx_1w_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below EMA50 AND weekly ADX > 25 (strong trend)
            elif close[i] < ema_50_1d_aligned[i] and adx_1w_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below EMA50 OR weekly trend weakens (ADX < 20)
            if close[i] < ema_50_1d_aligned[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above EMA50 OR weekly trend weakens (ADX < 20)
            if close[i] > ema_50_1d_aligned[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals