#!/usr/bin/env python3
"""
6h_RSI_Trend_Reversal_1d
Hypothesis: On 6h timeframe, RSI(14) combined with 1d trend filter provides
mean-reversion entries during pullbacks in strong trends. Works in both bull
and bear markets by trading with the higher timeframe trend. Uses RSI extremes
(>70 for short, <30 for long) with 1d EMA50 trend filter to avoid counter-
trend trades. Target: 20-40 trades/year per symbol.
"""

name = "6h_RSI_Trend_Reversal_1d"
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
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d trend: 50 EMA
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 6h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get aligned values
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # LONG: 1d uptrend + RSI oversold (<30)
            if uptrend and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + RSI overbought (>70)
            elif downtrend and rsi_val > 70:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (50) or trend turns down
            if rsi_val >= 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (50) or trend turns up
            if rsi_val <= 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals