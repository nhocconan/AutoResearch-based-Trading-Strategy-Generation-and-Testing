#!/usr/bin/env python3
"""
1D_RSI_Overbought_Oversold_With_Trend_Filter
Hypothesis: Buy when RSI crosses below 30 (oversold) in uptrend, sell when RSI crosses above 70 (overbought) in downtrend.
Uses 1D RSI with 1W EMA trend filter to avoid counter-trend trades. Works in both bull and bear markets by
only taking trades aligned with higher timeframe trend. Targets 10-25 trades/year to minimize fee drag.
"""
name = "1D_RSI_Overbought_Oversold_With_Trend_Filter"
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
    
    # Get 1D data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1D RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Get 1W data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1W EMA50 for trend direction
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 50)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI crosses below 30 (oversold) in uptrend (price above weekly EMA50)
            if (rsi_aligned[i] < 30 and rsi_aligned[i-1] >= 30 and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI crosses above 70 (overbought) in downtrend (price below weekly EMA50)
            elif (rsi_aligned[i] > 70 and rsi_aligned[i-1] <= 70 and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses above 50 (neutral) or trend reversal
            if rsi_aligned[i] > 50 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses below 50 (neutral) or trend reversal
            if rsi_aligned[i] < 50 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals