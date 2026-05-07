#!/usr/bin/env python3
"""
6h_KAMA_Trend_With_Volume_Confirmation
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) on 6h timeframe to capture adaptive trend direction, confirmed by volume spike (>1.5x 20-period average) and aligned with 12h EMA50 trend filter. Designed for low trade frequency (15-30/year) with adaptive trend-following that works in both bull and bear markets by requiring trend alignment and volatility-based confirmation.
"""

name = "6h_KAMA_Trend_With_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # KAMA parameters
    er_window = 10
    fast_sc = 2 / (2 + 1)  # 2/(fast+1)
    slow_sc = 2 / (30 + 1)  # 2/(slow+1)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_window))
    volatility = np.sum(np.abs(np.diff(close)), axis=1) if len(close) > 1 else 0
    # Handle volatility calculation properly
    volatility = np.array([np.sum(np.abs(np.diff(close[i-er_window+1:i+1])) if i >= er_window-1 else 0) 
                          for i in range(len(close))])
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    
    # Calculate smoothing constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend direction
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        trend_up = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] if i > 0 else False
        trend_down = ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Long: Price above KAMA, uptrend, volume spike
            if (price_above_kama and trend_up and vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA, downtrend, volume spike
            elif (price_below_kama and trend_down and vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below KAMA or trend turns down
            if not price_above_kama or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above KAMA or trend turns up
            if not price_below_kama or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals