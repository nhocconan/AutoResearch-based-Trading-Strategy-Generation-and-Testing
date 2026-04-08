#!/usr/bin/env python3
# 4h_1d_ema_rsi_pullback_v2
# Hypothesis: Pullback to 1d EMA with RSI oversold/overbought conditions on 4h timeframe.
# Uses 1d EMA as long-term trend filter and 4h RSI for mean-reversion entries.
# Works in both bull and bear markets by trading pullbacks to the trend.
# Reduced position size and tightened entry to reduce trade frequency and improve win rate.
# Target: 15-25 trades/year (60-100 total over 4 years).

name = "4h_1d_ema_rsi_pullback_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # 1d EMA trend filter (50-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h RSI (14-period)
    rsi_period = 14
    rsi = np.full(n, np.nan)
    if n >= rsi_period:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    # Start from sufficient lookback
    start_idx = max(50, rsi_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_1d_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI overbought or price breaks below EMA
            if rsi[i] > 70 or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI oversold or price breaks above EMA
            if rsi[i] < 30 or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: pullback to EMA with RSI oversold
            if close[i] >= ema_1d_aligned[i] and rsi[i] < 30:
                position = 1
                signals[i] = 0.20
            # Short: pullback to EMA with RSI overbought
            elif close[i] <= ema_1d_aligned[i] and rsi[i] > 70:
                position = -1
                signals[i] = -0.20
    
    return signals