#!/usr/bin/env python3
"""
6h_RegressionChannel_Retest
Hypothesis: In 6B markets, price often respects linear regression channels built from prior swing points.
We build a 50-bar linear regression channel on close prices, then look for retests of the channel
boundaries with volume confirmation and 12h trend alignment. Works in both trends (continuation) and
ranges (mean reversion at channels). Targets 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: using correct import from module

def _linear_regression_channel(arr, lookback):
    """Return upper and lower channel lines from linear regression of close prices."""
    n = len(arr)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    if lookback < 2:
        return upper, lower
    for i in range(lookback, n + 1):
        y = arr[i - lookback:i]
        x = np.arange(lookback)
        if np.all(np.isnan(y)):
            continue
        # Use only valid (non-nan) points
        mask = ~np.isnan(y)
        if np.sum(mask) < 2:
            continue
        x_valid = x[mask]
        y_valid = y[mask]
        # Solve least squares: y = mx + b
        A = np.vstack([x_valid, np.ones(len(x_valid))]).T
        m, b = np.linalg.lstsq(A, y_valid, rcond=None)[0]
        # Predict at last point of window
        y_end = m * (lookback - 1) + b
        # Calculate RMSE
        y_pred = m * x_valid + b
        rmse = np.sqrt(np.mean((y_valid - y_pred) ** 2))
        # Channel: ±1.5 * RMSE
        upper[i - 1] = y_end + 1.5 * rmse
        lower[i - 1] = y_end - 1.5 * rmse
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 60-bar linear regression channel on close (6-channel) ---
    lookback = 60
    ch_upper, ch_lower = _linear_regression_channel(close, lookback)
    
    # --- 12h EMA34 for trend filter ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    # EMA34
    ema_34_12h = np.zeros_like(close_12h)
    ema_34_12h[:] = np.nan
    if len(close_12h) >= 34:
        k = 2 / (34 + 1)
        ema_34_12h[33] = np.mean(close_12h[:34])
        for i in range(34, len(close_12h)):
            ema_34_12h[i] = close_12h[i] * k + ema_34_12h[i-1] * (1 - k)
    ema_34_aligned = align_ltf_to_htf(prices, df_12h, ema_34_12h)  # aligns 12h to 6m
    
    # --- Volume confirmation: current > 1.5x 20-bar average ---
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(lookback, 34)  # ensure channel and EMA ready
    
    for i in range(start_idx, n):
        if (np.isnan(ch_upper[i]) or np.isnan(ch_lower[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: retest of lower channel with volume spike and above 12h EMA34
            if low[i] <= ch_lower[i] * 1.001 and vol_spike[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: retest of upper channel with volume spike and below 12h EMA34
            elif high[i] >= ch_upper[i] * 0.999 and vol_spike[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: minimum 6 bars hold, then exit on channel retest or trend change
            if bars_since_entry >= 6:
                if high[i] >= ch_upper[i] * 0.999 or close[i] < ema_34_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25  # Hold during minimum period
        
        elif position == -1:
            # Exit: minimum 6 bars hold, then exit on channel retest or trend change
            if bars_since_entry >= 6:
                if low[i] <= ch_lower[i] * 1.001 or close[i] > ema_34_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25  # Hold during minimum period
    
    return signals

name = "6h_RegressionChannel_Retest"
timeframe = "6h"
leverage = 1.0