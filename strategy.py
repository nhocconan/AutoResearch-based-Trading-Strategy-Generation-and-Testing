#!/usr/bin/env python3
# 1D_1W_RSI_MeanReversion_TrendFilter
# Hypothesis: On 1d timeframe, enter long when weekly RSI < 30 and daily price > daily EMA50 (uptrend filter).
# Enter short when weekly RSI > 70 and daily price < daily EMA50 (downtrend filter).
# Uses weekly RSI for extreme mean-reversion signals and daily EMA50 for trend alignment.
# Target: 10-25 trades/year per symbol (40-100 total over 4 years).

name = "1D_1W_RSI_MeanReversion_TrendFilter"
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
    
    # Get weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(14)
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1w)
    avg_loss = np.zeros_like(close_1w)
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[:14])
    avg_loss[13] = np.mean(loss[:14])
    
    for i in range(14, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.concatenate([np.full(14, np.nan), rsi_1w[13:]])
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close > ema_50
    
    # Align weekly RSI to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(rsi_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: weekly RSI < 30 (oversold) + daily uptrend
            if rsi_1w_aligned[i] < 30 and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly RSI > 70 (overbought) + daily downtrend
            elif rsi_1w_aligned[i] > 70 and not trend_up[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly RSI > 50 (mean reversion complete) or trend fails
            if rsi_1w_aligned[i] > 50 or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly RSI < 50 (mean reversion complete) or trend fails
            if rsi_1w_aligned[i] < 50 or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals