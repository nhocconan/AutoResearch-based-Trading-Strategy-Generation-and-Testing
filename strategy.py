#!/usr/bin/env python3
"""
4h_RSI_Pullback_Trend_Filter_v1
Hypothesis: In strong trends (identified by 1d EMA34), RSI pullbacks offer high-probability 
entries with favorable risk-reward. Works in bull markets (buy pullbacks in uptrends) and 
bear markets (sell bounces in downtrends) by using the 1d trend as filter. 
Target: 25-50 trades per year (100-200 over 4 years) on 4h timeframe.
"""

name = "4h_RSI_Pullback_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # === 1D Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === RSI Calculation on 4h Close ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if trend data is invalid
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI pullback in uptrend (RSI < 40 and price above 1d EMA34)
            if rsi[i] < 40 and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI bounce in downtrend (RSI > 60 and price below 1d EMA34)
            elif rsi[i] > 60 and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI recovers to neutral or trend breaks
            if rsi[i] > 50 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI declines to neutral or trend breaks
            if rsi[i] < 50 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals