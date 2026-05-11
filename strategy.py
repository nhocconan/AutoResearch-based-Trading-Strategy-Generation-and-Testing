#!/usr/bin/env python3
"""
6h_Weekly_RSI_MeanReversion_v1
Hypothesis: Uses weekly RSI extremes (RSI > 70 or < 30) combined with 6h price action
to capture mean reversion moves. Weekly RSI > 70 indicates overbought conditions
on the weekly timeframe, suggesting potential short opportunities when price
rejects resistance. Weekly RSI < 30 indicates oversold conditions, suggesting
potential long opportunities when price finds support. Uses 6h RSI for entry
timing and volume confirmation to avoid false signals.
Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.
"""

name = "6h_Weekly_RSI_MeanReversion_v1"
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
    
    # === Weekly RSI for Regime ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # === 6h RSI for Entry Timing ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_6h = 100 - (100 / (1 + rs))
    
    # === Volume Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if weekly RSI is not available
        if np.isnan(rsi_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if 6h RSI or volume MA is not available
        if np.isnan(rsi_6h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: Weekly RSI < 30 (oversold) AND 6h RSI < 30 (oversold) AND volume > 20-day average
            if (rsi_1w_aligned[i] < 30 and rsi_6h[i] < 30 and volume[i] > vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: Weekly RSI > 70 (overbought) AND 6h RSI > 70 (overbought) AND volume > 20-day average
            elif (rsi_1w_aligned[i] > 70 and rsi_6h[i] > 70 and volume[i] > vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Weekly RSI > 50 (recovered from oversold) OR 6h RSI > 50
            if rsi_1w_aligned[i] > 50 or rsi_6h[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Weekly RSI < 50 (recovered from overbought) OR 6h RSI < 50
            if rsi_1w_aligned[i] < 50 or rsi_6h[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals