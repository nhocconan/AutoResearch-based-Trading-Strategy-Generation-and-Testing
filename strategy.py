#!/usr/bin/env python3
"""
4H_RSI_Divergence_Breakout_1D_Trend
Hypothesis: Combines RSI(2) divergence with 1-day trend confirmation for high-probability mean-reversion entries. RSI(2) < 10 signals oversold, > 90 overbought. Only takes trades in direction of 1D EMA50 trend. Designed for 4H timeframe to capture swing extremes in both bull and bear markets with low trade frequency.
"""

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
    
    # Calculate RSI(2) - fast RSI for extreme conditions
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / 2
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1D data for trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1D EMA50 for trend
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for RSI and EMA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema50_val = ema50_1d_aligned[i]
        
        if position == 0:
            # Long: RSI(2) oversold (<10) AND price above 1D EMA50 (uptrend)
            if rsi_val < 10 and close[i] > ema50_val:
                signals[i] = size
                position = 1
            # Short: RSI(2) overbought (>90) AND price below 1D EMA50 (downtrend)
            elif rsi_val > 90 and close[i] < ema50_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI(2) returns to neutral (>50) OR price crosses below EMA50
            if rsi_val > 50 or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI(2) returns to neutral (<50) OR price crosses above EMA50
            if rsi_val < 50 or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4H_RSI_Divergence_Breakout_1D_Trend"
timeframe = "4h"
leverage = 1.0