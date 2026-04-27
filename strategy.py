#!/usr/bin/env python3
"""
4h_Volume_Weighted_RSI_12hTrend
Hypothesis: Combine volume-weighted RSI(14) with 12h EMA50 trend filter to capture 
momentum extremes in the direction of higher timeframe trend. Volume weighting 
reduces false signals during low-volume periods. Designed for 15-25 trades/year 
per symbol with clear entry/exit rules to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume-weighted RSI calculation
    # Calculate price changes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Weight gains and losses by volume
    vol_weight = volume / (np.mean(volume) + 1e-10)  # Normalize volume
    weighted_gain = gain * vol_weight
    weighted_loss = loss * vol_weight
    
    # Calculate smoothed averages with proper min_periods
    avg_gain = pd.Series(weighted_gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(weighted_loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate RSI
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for RSI calculation
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if trend data not ready
        if np.isnan(ema_50_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema_50_val = ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) and price above 12h EMA50 (uptrend)
            if rsi_val < 30 and close[i] > ema_50_val:
                signals[i] = size
                position = 1
            # Short: RSI > 70 (overbought) and price below 12h EMA50 (downtrend)
            elif rsi_val > 70 and close[i] < ema_50_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (momentum fading) or trend change
            if rsi_val > 50 or close[i] < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI < 50 (momentum fading) or trend change
            if rsi_val < 50 or close[i] > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Volume_Weighted_RSI_12hTrend"
timeframe = "4h"
leverage = 1.0