#!/usr/bin/env python3
"""
#100988 - 12h_RSI_21_50_Reversal
Hypothesis: Mean-reversion using RSI(21) oversold/overbought levels with 50 as neutral line. Works in both bull and bear markets by capturing short-term reversals within larger trends. Uses 1w trend filter to avoid counter-trend trades. Target: 15-25 trades/year to minimize fee drag. Uses discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate RSI(21) on close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/21, adjust=False, min_periods=21).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/21, adjust=False, min_periods=21).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(rsi[i]) or np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long condition: RSI < 30 (oversold) and price above 1w EMA50 (uptrend filter)
        if rsi[i] < 30 and close[i] > ema50_1w_aligned[i]:
            signals[i] = 0.25
            position = 1
        # Short condition: RSI > 70 (overbought) and price below 1w EMA50 (downtrend filter)
        elif rsi[i] > 70 and close[i] < ema50_1w_aligned[i]:
            signals[i] = -0.25
            position = -1
        # Exit conditions: RSI crosses back through 50 (mean reversion complete)
        elif position == 1 and rsi[i] > 50:
            signals[i] = 0.0
            position = 0
        elif position == -1 and rsi[i] < 50:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_RSI_21_50_Reversal"
timeframe = "12h"
leverage = 1.0