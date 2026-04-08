#!/usr/bin/env python3
# 6h_1d_rsi_mean_reversion_v1
# Hypothesis: Mean reversion on 6h timeframe using RSI(14) with 1-day trend filter.
# Long when RSI < 30 and 1-day close > 1-day SMA(50) (uptrend filter).
# Short when RSI > 70 and 1-day close < 1-day SMA(50) (downtrend filter).
# Uses 6h RSI for entry timing and 1-day trend to avoid counter-trend trades.
# Designed for 50-150 total trades over 4 years (12-37/year) with position size 0.25.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_rsi_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # 6h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure SMA(50) and RSI are ready
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(sma50_1d_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (mean reversion complete) or trend reverses
            if rsi[i] > 50 or close[i] < sma50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (mean reversion complete) or trend reverses
            if rsi[i] < 50 or close[i] > sma50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI < 30 (oversold) and uptrend
            if rsi[i] < 30 and close[i] > sma50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI > 70 (overbought) and downtrend
            elif rsi[i] > 70 and close[i] < sma50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals