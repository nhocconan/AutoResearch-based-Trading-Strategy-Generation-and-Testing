#!/usr/bin/env python3
"""
1d_1w_RSI_Momentum_With_TrendFilter_v1
Concept: Use weekly RSI momentum on daily timeframe with trend filter.
- Long when daily RSI(14) > 50 and weekly RSI(14) > 50 and price above weekly EMA(50)
- Short when daily RSI(14) < 50 and weekly RSI(14) < 50 and price below weekly EMA(50)
- Exit when RSI crosses back below/above 50
- Sizing 0.25 to manage drawdown
- Works in bull (momentum continuation) and bear (mean reversion via RSI extremes)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_RSI_Momentum_With_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend and momentum
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly: RSI(14) and EMA(50) for trend filter ===
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(14)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss != 0, avg_loss, 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Weekly EMA(50) for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Daily: RSI(14) for entry signal ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss != 0, avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for weekly EMA50
    
    for i in range(start_idx, n):
        # Get values
        rsi_val = rsi[i]
        rsi_1w_val = rsi_1w_aligned[i]
        ema50_1w_val = ema50_1w_aligned[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(rsi_1w_val) or np.isnan(ema50_1w_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Daily RSI > 50, Weekly RSI > 50, and price above weekly EMA50
            if rsi_val > 50 and rsi_1w_val > 50 and close_val > ema50_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: Daily RSI < 50, Weekly RSI < 50, and price below weekly EMA50
            elif rsi_val < 50 and rsi_1w_val < 50 and close_val < ema50_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Daily RSI crosses back below 50
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Daily RSI crosses back above 50
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals