#!/usr/bin/env python3
"""
1h_4d_roc_extreme_v1
Strategy: 1h ROC(12) extreme reversal with 4h trend filter
Timeframe: 1h
Leverage: 1.0
Hypothesis: Uses ROC(12) extremes (>5% for short, <-5% for long) on 1h combined with 4h EMA50 trend filter. Designed to capture short-term mean reversals while following the 4h trend. Works in both bull/bear markets by following the higher timeframe trend. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_roc_extreme_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load higher timeframe data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1h ROC(12)
    roc = np.zeros_like(close)
    roc[12:] = (close[12:] - close[:-12]) / close[:-12] * 100.0
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(roc[i]) or np.isnan(ema_50_4h_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        price_close = close[i]
        
        # Trend filter
        uptrend_4h = price_close > ema_50_4h_aligned[i]
        downtrend_4h = price_close < ema_50_4h_aligned[i]
        
        # ROC extreme conditions
        roc_overbought = roc[i] > 5.0
        roc_oversold = roc[i] < -5.0
        
        # Long: ROC oversold in uptrend
        long_signal = roc_oversold and uptrend_4h
        
        # Short: ROC overbought in downtrend
        short_signal = roc_overbought and downtrend_4h
        
        # Exit when ROC returns to neutral zone
        exit_long = position == 1 and roc[i] > -1.0
        exit_short = position == -1 and roc[i] < 1.0
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Uses ROC(12) extremes (>5% for short, <-5% for long) on 1h combined with 4h EMA50 trend filter. Designed to capture short-term mean reversals while following the 4h trend. Works in both bull/bear markets by following the higher timeframe trend. Target: 60-150 total trades over 4 years.