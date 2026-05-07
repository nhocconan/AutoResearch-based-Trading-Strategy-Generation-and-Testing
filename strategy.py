#!/usr/bin/env python3
"""
6h_Volume_Weighted_RSI_Pullback
Hypothesis: In strong trends (price > weekly EMA20), buy pullbacks when volume-weighted RSI < 30; sell rallies when price < weekly EMA20 and volume-weighted RSI > 70. 
Volume weighting filters for institutional participation. Works in bull/bear via weekly trend filter.
Target: 15-25 trades/year to minimize fee drag.
"""

name = "6h_Volume_Weighted_RSI_Pullback"
timeframe = "6h"
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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate volume-weighted RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volume-weighted gain/loss
    vol_weight = volume / (np.mean(volume) + 1e-9)
    vol_gain = gain * vol_weight
    vol_loss = loss * vol_weight
    
    # Smoothed averages
    avg_gain = pd.Series(vol_gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(vol_loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        trend_up = close[i] > ema_20_1w_aligned[i]
        trend_down = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: pullback in uptrend when volume-weighted RSI < 30
            if trend_up and rsi[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: rally in downtrend when volume-weighted RSI > 70
            elif trend_down and rsi[i] > 70:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (momentum shift) or trend turns down
            if rsi[i] > 50 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 50 (momentum shift) or trend turns up
            if rsi[i] < 50 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals