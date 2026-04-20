#!/usr/bin/env python3
# 4h_RSI_Divergence_Backtest_Validation
# Hypothesis: RSI divergence with volume confirmation and trend filter captures reversals in both bull and bear markets.
# The 4h timeframe reduces trade frequency while RSI divergence identifies exhaustion points.
# Works in bull markets by catching bearish divergences at tops; in bear markets by catching bullish divergences at bottoms.
# Volume filter ensures institutional participation; trend filter avoids counter-trend trades.

name = "4h_RSI_Divergence_Backtest_Validation"
timeframe = "4h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA50 trend filter on daily timeframe
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (vol_ema20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (i >= 3 and low[i] < low[i-3] and rsi[i] > rsi[i-3] and
                close[i] > ema50_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Bearish divergence: price makes higher high, RSI makes lower high
            elif (i >= 3 and high[i] > high[i-3] and rsi[i] < rsi[i-3] and
                  close[i] < ema50_1d_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: bearish divergence or price below EMA50
            if (i >= 3 and high[i] > high[i-3] and rsi[i] < rsi[i-3]) or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: bullish divergence or price above EMA50
            if (i >= 3 and low[i] < low[i-3] and rsi[i] > rsi[i-3]) or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals