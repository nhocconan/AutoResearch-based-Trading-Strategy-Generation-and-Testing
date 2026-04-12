#!/usr/bin/env python3
"""
6h_1d_1w_Altitude_Gradient_v1
Hypothesis: Capture momentum persistence by measuring the slope (gradient) of multi-timeframe moving averages. In both bull and bear markets, strong trends exhibit aligned upward/downward slopes across 6h, 1d, and 1w timeframes. Uses normalized slope comparison to avoid scale issues and reduce false signals. Target: 80-120 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Altitude_Gradient_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6H EMA(21) FOR GRADIENT ===
    if len(close) >= 21:
        alpha_6h = 2.0 / (21 + 1)
        ema_6h = np.zeros_like(close)
        ema_6h[0] = close[0]
        for i in range(1, len(close)):
            ema_6h[i] = alpha_6h * close[i] + (1 - alpha_6h) * ema_6h[i-1]
    else:
        ema_6h = np.full_like(close, np.nan)
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA(21)
    alpha_1d = 2.0 / (21 + 1)
    ema_1d = np.zeros_like(close_1d)
    ema_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema_1d[i] = alpha_1d * close_1d[i] + (1 - alpha_1d) * ema_1d[i-1]
    
    # === WEEKLY DATA ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA(21)
    alpha_1w = 2.0 / (21 + 1)
    ema_1w = np.zeros_like(close_1w)
    ema_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema_1w[i] = alpha_1w * close_1w[i] + (1 - alpha_1w) * ema_1w[i-1]
    
    # === GRADIENT CALCULATION (3-period slope) ===
    def calculate_gradient(series, window=3):
        """Calculate normalized slope over window periods"""
        if len(series) < window:
            return np.full_like(series, np.nan)
        grad = np.full_like(series, np.nan)
        for i in range(window-1, len(series)):
            if i >= window:
                slope = (series[i] - series[i-window]) / window
                # Normalize by price level to make comparable across timeframes
                grad[i] = slope / series[i-window] if series[i-window] != 0 else 0
        return grad
    
    grad_6h = calculate_gradient(ema_6h, 3)
    grad_1d = calculate_gradient(ema_1d, 3)
    grad_1w = calculate_gradient(ema_1w, 3)
    
    # Align HTF gradients to 6h timeframe
    grad_1d_aligned = align_htf_to_ltf(prices, df_1d, grad_1d)
    grad_1w_aligned = align_htf_to_ltf(prices, df_1w, grad_1w)
    
    # Volume confirmation (20-period average)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        vol_avg[i] = vol_sum / vol_count if vol_count > 0 else 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if gradients not available
        if (np.isnan(grad_6h[i]) or np.isnan(grad_1d_aligned[i]) or 
            np.isnan(grad_1w_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Gradient alignment conditions
        # Strong uptrend: all gradients positive and increasing
        strong_uptrend = (grad_6h[i] > 0) and (grad_1d_aligned[i] > 0) and (grad_1w_aligned[i] > 0) and \
                         (grad_6h[i] > grad_1d_aligned[i] * 0.5) and (grad_1d_aligned[i] > grad_1w_aligned[i] * 0.5)
        
        # Strong downtrend: all gradients negative and decreasing
        strong_downtrend = (grad_6h[i] < 0) and (grad_1d_aligned[i] < 0) and (grad_1w_aligned[i] < 0) and \
                           (abs(grad_6h[i]) > abs(grad_1d_aligned[i]) * 0.5) and (abs(grad_1d_aligned[i]) > abs(grad_1w_aligned[i]) * 0.5)
        
        # Exit conditions: gradient divergence or loss of momentum
        exit_long = (grad_6h[i] <= 0) or (grad_1d_aligned[i] <= 0) or not vol_confirm
        exit_short = (grad_6h[i] >= 0) or (grad_1d_aligned[i] >= 0) or not vol_confirm
        
        if strong_uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif strong_downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals