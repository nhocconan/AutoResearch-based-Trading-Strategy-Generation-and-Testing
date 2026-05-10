#!/usr/bin/env python3
"""
1h_Trend_Reversal_with_Volume_Confirmation
Hypothesis: Mean-reversion strategy that buys when price closes below Bollinger Lower Band with volume confirmation,
and sells when price closes above Bollinger Upper Band with volume confirmation. Uses 4h trend filter to avoid
trading against the higher timeframe trend. Works in both bull and bear markets by capitalizing on short-term
reversions within larger trends. Low trade frequency due to strict entry conditions.
"""

name = "1h_Trend_Reversal_with_Volume_Confirmation"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mpt_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h close for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend direction
    ema50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema50_4h[49] = np.mean(close_4h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema50_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema50_4h[i-1]
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Bollinger Bands (20, 2) on 1h close
    bb_length = 20
    bb_mult = 2.0
    basis = np.full(n, np.nan)
    dev = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n >= bb_length:
        # Calculate SMA20
        sma = np.full(n, np.nan)
        sma[bb_length-1] = np.mean(close[:bb_length])
        for i in range(bb_length, n):
            sma[i] = np.mean(close[i-bb_length+1:i+1])
        
        # Calculate standard deviation
        variance = np.full(n, np.nan)
        for i in range(bb_length-1, n):
            variance[i] = np.mean((close[i-bb_length+1:i+1] - sma[i]) ** 2)
        
        basis = sma
        dev = bb_mult * np.sqrt(variance)
        upper = basis + dev
        lower = basis - dev
    
    # Volume confirmation: current volume > 1.5x average volume (20-period)
    vol_sma20 = np.full(n, np.nan)
    if n >= 20:
        vol_sma20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_sma20[i] = (vol_sma20[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, bb_length, 20)  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(basis[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_sma20[i]
        
        if position == 0:
            # Long: price closes below lower band AND 4h trend is up (avoid fighting downtrend)
            if close[i] < lower[i] and close[i] > ema50_4h_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: price closes above upper band AND 4h trend is down (avoid fighting uptrend)
            elif close[i] > upper[i] and close[i] < ema50_4h_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price crosses back above basis (mean reversion complete) or trend changes
            if close[i] > basis[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price crosses back below basis or trend changes
            if close[i] < basis[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals