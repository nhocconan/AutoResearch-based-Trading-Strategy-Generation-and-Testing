#!/usr/bin/env python3
"""
4h_Trend_Reversal_With_Confluence_v1
Hypothesis: Use 4-hour timeframe with a contrarian mean-reversion signal based on RSI extremes,
filtered by higher-timeframe trend (1-day EMA) and volume confirmation. Designed to capture
reversals in both bull and bear markets by entering when momentum is exhausted but the
underlying trend remains intact. Uses tight conditions to limit trades and avoid fee drag.
"""

name = "4h_Trend_Reversal_With_Confluence_v1"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) for mean-reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1-day trend filter: EMA(34) of daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30), price above 1-day EMA (uptrend filter), volume confirmation
            if rsi[i] < 30 and close[i] > ema_1d_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70), price below 1-day EMA (downtrend filter), volume confirmation
            elif rsi[i] > 70 and close[i] < ema_1d_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI returns to neutral (50) or trend changes
            if rsi[i] >= 50 or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI returns to neutral (50) or trend changes
            if rsi[i] <= 50 or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals