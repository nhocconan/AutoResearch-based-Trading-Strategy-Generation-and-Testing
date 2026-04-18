#!/usr/bin/env python3
"""
1d_1w_RSI_Divergence_Pivot_Reversion
Hypothesis: Mean reversion at weekly pivots with daily RSI divergence. In bull markets, buy RSI oversold near weekly S1/S2; in bear markets, sell RSI overbought near weekly R1/R2. Uses weekly pivot levels as dynamic support/resistance and RSI divergence for exhaustion signals. Works in both regimes by fading extremes at key levels. Targets 15-25 trades/year via tight confluence of RSI extremes (<30/>70) + pivot proximity + divergence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly OHLC for pivot points (using prior week's data)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Prior week's OHLC (completed week)
    prev_high_w = np.roll(high_w, 1)
    prev_low_w = np.roll(low_w, 1)
    prev_close_w = np.roll(close_w, 1)
    prev_high_w[0] = high_w[0]
    prev_low_w[0] = low_w[0]
    prev_close_w[0] = close_w[0]
    
    # Weekly pivot levels (standard floor trader pivots)
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3
    r1_w = 2 * pivot_w - prev_low_w
    s1_w = 2 * pivot_w - prev_high_w
    r2_w = pivot_w + (prev_high_w - prev_low_w)
    s2_w = pivot_w - (prev_high_w - prev_low_w)
    
    # Daily RSI (14-period) for divergence
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI divergence detection: price makes new high/low but RSI does not
    # Bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    lookback = 5
    bull_div = np.zeros(n, dtype=bool)
    bear_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Bullish divergence
        if low[i] < low[i-lookback] and rsi[i] > rsi[i-lookback]:
            # Check if this is meaningful divergence
            if rsi[i] < 40:  # Only in oversold territory
                bull_div[i] = True
        # Bearish divergence
        if high[i] > high[i-lookback] and rsi[i] < rsi[i-lookback]:
            if rsi[i] > 60:  # Only in overbought territory
                bear_div[i] = True
    
    # Align weekly pivot levels to daily timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # RSI needs 14 periods
    
    for i in range(start_idx, n):
        # Skip if pivot data not available
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or np.isnan(r2_w_aligned[i]) or 
            np.isnan(s2_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: bullish divergence near weekly support
            if (bull_div[i] and 
                (low[i] <= s2_w_aligned[i] * 1.02 or low[i] <= s1_w_aligned[i] * 1.02) and
                rsi[i] < 35):
                signals[i] = 0.25
                position = 1
            # Short setup: bearish divergence near weekly resistance
            elif (bear_div[i] and 
                  (high[i] >= r2_w_aligned[i] * 0.98 or high[i] >= r1_w_aligned[i] * 0.98) and
                  rsi[i] > 65):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish divergence or RSI overbought
            if bear_div[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish divergence or RSI oversold
            if bull_div[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_RSI_Divergence_Pivot_Reversion"
timeframe = "1d"
leverage = 1.0