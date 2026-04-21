#!/usr/bin/env python3
"""
4h_RSI_Pullback_Trend_Filter_V1
Hypothesis: In trending markets (4h), RSI pullbacks to the 50 level provide high-probability entries with the trend. Uses 12h EMA200 as trend filter to avoid counter-trend trades. Works in bull/bear by only taking trades aligned with higher timeframe trend. Low frequency (~25 trades/year) minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    # Calculate EMA200 on 12h close
    close_12h = df_12h['close'].values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # 4h RSI(14)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if EMA not ready
        if np.isnan(ema200_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema200 = ema200_12h_aligned[i]
        
        if position == 0:
            # Long: price above 12h EMA200 + RSI pulls back to 50 from above
            if price > ema200 and 45 <= rsi_val <= 55 and rsi[i-1] > 55:
                signals[i] = 0.25
                position = 1
            # Short: price below 12h EMA200 + RSI pulls back to 50 from below
            elif price < ema200 and 45 <= rsi_val <= 55 and rsi[i-1] < 45:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI reaches overbought or trend breaks
            if rsi_val >= 70 or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI reaches oversold or trend breaks
            if rsi_val <= 30 or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Pullback_Trend_Filter_V1"
timeframe = "4h"
leverage = 1.0