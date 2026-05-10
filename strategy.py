#!/usr/bin/env python3
# 1d_Weekly_Trend_Follower
# Hypothesis: Weekly trend direction determines the long-term bias, and daily pullbacks
# to the 21-day EMA in line with that trend offer high-probability entries in both bull
# and bear markets. Uses weekly EMA21 for trend filter and daily RSI pullback for entry.
# Designed for very low trade frequency (~10-20/year) to minimize fee drag and avoid
# whipsaws in ranging markets.

name = "1d_Weekly_Trend_Follower"
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
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA21 for trend (smooth, lag-appropriate)
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily RSI(14) for pullback entries
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 21)  # need enough history for weekly EMA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(ema_21_1w_aligned[i]) or np.isnan(rsi_values[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly uptrend (price above weekly EMA21) and daily RSI oversold (<30)
            if close[i] > ema_21_1w_aligned[i] and rsi_values[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend (price below weekly EMA21) and daily RSI overbought (>70)
            elif close[i] < ema_21_1w_aligned[i] and rsi_values[i] > 70:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend turns down or RSI overbought
            if close[i] < ema_21_1w_aligned[i] or rsi_values[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend turns up or RSI oversold
            if close[i] > ema_21_1w_aligned[i] or rsi_values[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals