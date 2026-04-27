#!/usr/bin/env python3
"""
4h_Constitutional_Conservative_Edge
Hypothesis: Combines 1-day volatility breakout (ATR-based) with 4-hour momentum confirmation and volume filter. Designed for low trade frequency (target 15-25/year) to minimize fee drag while capturing strong directional moves in both bull and bear markets. Uses conservative position sizing and strict entry conditions.
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
    
    # Get 1d data for volatility calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day ATR(14) for volatility breakout
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.inf], tr2])  # First TR undefined
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1-day EMA(50) for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4-hour momentum (ROC 3-period)
    roc_period = 3
    roc = np.zeros_like(close)
    roc[roc_period:] = (close[roc_period:] - close[:-roc_period]) / close[:-roc_period] * 100
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for ATR, EMA, ROC, and volume
    start_idx = max(50, 20, roc_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(roc[i])):
            signals[i] = 0.0
            continue
        
        atr_val = atr_14_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        roc_val = roc[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: volatility breakout up + positive momentum + uptrend + volume
            if (close[i] > close[i-1] + 0.5 * atr_val and  # Break above prior close + 0.5*ATR
                roc_val > 0.2 and                          # Positive momentum
                close[i] > ema_trend and                   # Above daily EMA50
                vol_ok):                                   # Volume confirmation
                signals[i] = size
                position = 1
            # Short: volatility breakout down + negative momentum + downtrend + volume
            elif (close[i] < close[i-1] - 0.5 * atr_val and  # Break below prior close - 0.5*ATR
                  roc_val < -0.2 and                         # Negative momentum
                  close[i] < ema_trend and                   # Below daily EMA50
                  vol_ok):                                   # Volume confirmation
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: momentum dies or trend breaks
            if roc_val < 0 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: momentum dies or trend breaks
            if roc_val > 0 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Constitutional_Conservative_Edge"
timeframe = "4h"
leverage = 1.0