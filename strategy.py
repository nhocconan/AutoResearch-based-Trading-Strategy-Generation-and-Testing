#!/usr/bin/env python3
# 1d_RSI_Extreme_1wTrend_Filter
# Hypothesis: RSI extremes on daily timeframe (RSI<30 or >70) combined with weekly trend filter
# (price above/below 200 EMA) provides high-probability mean reversion entries in ranging markets
# and trend continuation in strong trends. Weekly EMA200 filter ensures we only take trades
# in the direction of the higher timeframe trend, reducing false signals during chop.
# Target: 10-25 trades/year on 1d timeframe.

name = "1d_RSI_Extreme_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    ema200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA200 to daily timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily RSI calculation
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to RMA)
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI (14) + weekly EMA200 (200)
    start_idx = max(14, 200)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) AND price above weekly EMA200 (uptrend)
            if rsi[i] < 30 and close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) AND price below weekly EMA200 (downtrend)
            elif rsi[i] > 70 and close[i] < ema200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion) OR price below weekly EMA200 (trend change)
            if rsi[i] > 50 or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion) OR price above weekly EMA200 (trend change)
            if rsi[i] < 50 or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals